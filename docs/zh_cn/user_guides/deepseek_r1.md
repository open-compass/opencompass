# 强推理模型评测教程

OpenCompass提供针对DeepSeek R1系列推理模型的评测教程（数学数据集）。

- 在模型层面，我们建议使用Sampling方式，以减少因为Greedy评测带来的大量重复
- 在数据集层面，我们对数据量较小的评测基准，使用多次评测并取平均的方式。
- 在答案验证层面，为了减少基于规则评测带来的误判，我们统一使用基于LLM验证的方式进行评测。

## 安装和准备

请按OpenCompass安装教程进行安装。

## 构建评测配置

我们在 `example/eval_deepseek_r1.py` 中提供了示例配置，以下对评测配置进行解读

### 评测配置解读

#### 1. 数据集与验证器配置

```python
# 支持多运行次数的数据集配置（示例）
from opencompass.configs.datasets.aime2024.aime2024_llmverify_repeat8_gen_e8fcee import aime2024_datasets

datasets = sum(
    (v for k, v in locals().items() if k.endswith('_datasets')),
    [],
)

# 设置LLM验证器， 用户需事先通过LMDeploy/vLLM/SGLang等工具启动API 评测服务器，或者直接使用兼容OpenAI标准接口的模型服务
verifier_cfg = dict(
    abbr='qwen2-5-32B-Instruct',
    type=OpenAISDK,
    path='Qwen/Qwen2.5-32B-Instruct',  # 需替换实际路径
    key='YOUR_API_KEY',  # 需替换真实API Key
    openai_api_base=['http://your-api-endpoint'],  # 需替换API地址
    query_per_second=16,
    batch_size=1024,
    temperature=0.001,
    max_out_len=16384
)

# 应用验证器到所有数据集
for item in datasets:
    if 'judge_cfg' in item['eval_cfg']['evaluator']:
        item['eval_cfg']['evaluator']['judge_cfg'] = verifier_cfg
```

#### 2. 模型配置

我们提供了基于LMDeploy作为推理后端的评测示例，用户可以通过修改path（即HF路径）

```python
# LMDeploy模型配置示例
models = [
    dict(
        type=TurboMindModelwithChatTemplate,
        abbr='deepseek-r1-distill-qwen-7b-turbomind',
        path='deepseek-ai/DeepSeek-R1-Distill-Qwen-7B',
        engine_config=dict(session_len=32768, max_batch_size=128, tp=1),
        gen_config=dict(
            do_sample=True,
            temperature=0.6,
            top_p=0.95,
            max_new_tokens=32768
        ),
        max_seq_len=32768,
        batch_size=64,
        run_cfg=dict(num_gpus=1),
        pred_postprocessor=dict(type=extract_non_reasoning_content)
    ),
    # 可扩展14B/32B配置...
]
```

#### 3. 评估流程配置

```python
# 推理配置
infer = dict(
    partitioner=dict(type=NumWorkerPartitioner, num_worker=1),
    runner=dict(type=LocalRunner, task=dict(type=OpenICLInferTask))
    
# 评估配置
eval = dict(
    partitioner=dict(type=NaivePartitioner, n=8),
    runner=dict(type=LocalRunner, task=dict(type=OpenICLEvalTask)))
```

#### 4. 结果汇总配置

```python
# 多运行结果平均配置
summary_groups = [
    {
        'name': 'AIME2024-Aveage8',
        'subsets':[[f'aime2024-run{idx}', 'accuracy'] for idx in range(8)]
    },
    # 其他数据集平均配置...
]

summarizer = dict(
    dataset_abbrs=[
        ['AIME2024-Aveage8', 'naive_average'],
        # 其他数据集指标...
    ],
    summary_groups=summary_groups
)

# 工作目录设置
work_dir = "outputs/deepseek_r1_reasoning"
```

## 执行评测

### 场景1：模型1卡加载，数据1个worker评测，共使用1个GPU

```bash
opencompass example/eval_deepseek_r1.py --debug --dump-eval-details
```

评测日志会在命令行输出。

### 场景2：模型1卡加载，数据8个worker评测，共使用8个GPU

需要修改配置文件中的infer配置，将num_worker设置为8

```python
# 推理配置
infer = dict(
    partitioner=dict(type=NumWorkerPartitioner, num_worker=1),
    runner=dict(type=LocalRunner, task=dict(type=OpenICLInferTask))
```

同时评测命令去掉`--debug`参数

```bash
opencompass example/eval_deepseek_r1.py --dump-eval-details
```

此模式下，OpenCompass将使用多线程启动`$num_worker`个任务，命令行不展示具体日志，具体的评测日志将会在`$work_dir`下中展示。

### 场景3：模型2卡加载，数据4个worker评测，共使用8个GPU

需要注意模型配置中，`run_cfg`中的`num_gpus`需要设置为2(如使用推理后端，则推理后端的参数也需要同步修改，比如LMDeploy中的tp需要设置为2)，同时修改`infer`配置中的`num_worker`为4

```python
models += [
    dict(
        type=TurboMindModelwithChatTemplate,
        abbr='deepseek-r1-distill-qwen-14b-turbomind',
        path='deepseek-ai/DeepSeek-R1-Distill-Qwen-14B',
        engine_config=dict(session_len=32768, max_batch_size=128, tp=2),
        gen_config=dict(
                        do_sample=True,
                        temperature=0.6,
                        top_p=0.95,
                        max_new_tokens=32768),
        max_seq_len=32768,
        max_out_len=32768,
        batch_size=128,
        run_cfg=dict(num_gpus=2),
        pred_postprocessor=dict(type=extract_non_reasoning_content)
    ),
]
```

```python
# 推理配置
infer = dict(
    partitioner=dict(type=NumWorkerPartitioner, num_worker=4),
    runner=dict(type=LocalRunner, task=dict(type=OpenICLInferTask))
```

### 评测结果

评测结果展示如下：

```bash
dataset                             version    metric         mode    deepseek-r1-distill-qwen-7b-turbomind                                                                                                       ----------------------------------  ---------  -------------  ------  ---------------------------------------                                                                                                     MATH                                -          -              -                                         AIME2024-Aveage8                    -          naive_average  gen     56.25     

```

## 性能基线参考

由于模型使用Sampling进行解码，同时AIME数据量较小，使用8次评测取平均情况下，仍会出现1-3分的性能抖动

| 模型                         | 数据集   | 指标     | 数值 |
| ---------------------------- | -------- | -------- | ---- |
| DeepSeek-R1-Distill-Qwen-7B  | AIME2024 | Accuracy | 56.3 |
| DeepSeek-R1-Distill-Qwen-14B | AIME2024 | Accuracy | 74.2 |
| DeepSeek-R1-Distill-Qwen-32B | AIME2024 | Accuracy | 74.2 |
