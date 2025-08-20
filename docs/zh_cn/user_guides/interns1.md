# Intern-S1评测教程

OpenCompass现已提供评测Intern-S1所需的相关模型配置与数据集配置。请顺序执行下列步骤来启动对Intern-S1的评测。

## 模型下载与部署

Intern-S1的模型权重现已开源，请从[Huggingface](https://huggingface.co/internlm/Intern-S1)获取。
完成模型下载后，推荐将其部署为API服务形式进行调用。可根据[此页面](https://github.com/InternLM/Intern-S1/blob/main/README.md#Serving)上提供的LMdeploy/vLLM/sglang形式进行部署。

## 评测配置

### 模型配置

我们在`opencompass/configs/models/interns1/intern_s1.py`中提供了OpenAISDK形式调用模型的配置示例，请根据你的需求进行相应更改。

```python
models = [
    dict(
        abbr="intern-s1",
        key="YOUR_API_KEY", # 在此处填写模型服务的API KEY
        openai_api_base="YOUR_API_BASE", # 在此处填写模型服务的API BASE
        type=OpenAISDK,
        path="internlm/Intern-S1",
        temperature=0.7,
        meta_template=api_meta_template,
        query_per_second=1,
        batch_size=8,
        max_out_len=64000,
        max_seq_len=65536,
        openai_extra_kwargs={
            'top_p': 0.95,
        },
        retry=10,
        extra_body={
            "chat_template_kwargs": {"enable_thinking": True} # 基于vllm或sglang部署服务后通过该开关来调控模型的思考模式
        },
        pred_postprocessor=dict(type=extract_non_reasoning_content), # 开启思考模式后可添加此配置来在Eval时去除Thinking内容
    ),
]
```

### 数据集配置

我们在`examples/eval_bench_intern_s1.py`中提供了评测Intern-S1所使用的相关数据集配置。你也可以根据需要自行添加其他数据集。

此外，你还需在该配置文件中添加LLM Judger的配置，示例如下：

```python
judge_cfg = dict(
    abbr='YOUR_JUDGE_MODEL',
    type=OpenAISDK,
    path='YOUR_JUDGE_MODEL_PATH',
    key='YOUR_API_KEY',
    openai_api_base='YOUR_API_BASE',
    meta_template=dict(
        round=[
            dict(role='HUMAN', api_role='HUMAN'),
            dict(role='BOT', api_role='BOT', generate=True),
        ]),
    query_per_second=1,
    batch_size=1,
    temperature=0.001,
    max_out_len=8192,
    max_seq_len=32768,
    mode='mid',
)
```

## 启动评测

完成上述配置后，在命令行输入下面的指令启动评测：

```bash
opencompass examples/eval_bench_intern_s1.py
```
