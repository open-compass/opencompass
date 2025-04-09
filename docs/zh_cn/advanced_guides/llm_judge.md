# LLM 作为评判器

## 简介

GenericLLMEvaluator组件特别适用于那些难以通过规则式方法（如正则表达式）进行完美判断的场景，例如：

- 模型不输出选项标识而只输出选项内容的情况
- 需要事实性判断的数据集
- 需要复杂理解和推理的开放式回答
- 需要设计大量规则的判断

OpenCompass提供了GenericLLMEvaluator组件来实现LLM作为评判器的评估。

## 数据集格式

用于LLM评判的数据集应该是JSON Lines (.jsonl)或CSV格式。每个条目至少应包含：

- 问题或任务
- 参考答案或标准答案
- (模型的预测将在评估过程中生成)

JSONL格式示例：

```json
{"problem": "法国的首都是什么？", "answer": "巴黎"}
```

CSV格式示例：

```csv
problem,answer
"法国的首都是什么？","巴黎"
```

## 配置说明

### 基于命令行使用LLM进行评估

OpenCompass中部分数据集已经包含了LLM评判器的配置。
你需要使用一个模型服务（如OpenAI或DeepSeek官方提供的API）或本地使用LMDeploy、vLLM、SGLang等工具启动一个模型服务。

然后，你可以通过以下命令设置相关评估服务的环境变量，并对模型进行评估：

```bash
export OC_JUDGE_MODEL=Qwen/Qwen2.5-32B-Instruct
export OC_JUDGE_API_KEY=sk-1234
export OC_JUDGE_API_BASE=http://172.30.56.1:4000/v1 
```

注意，默认情况下，OpenCompass会使用这三个环境变量，但如果你使用了基于配置文件的方式配置评估服务，这三个环境变量将不会生效。

### 基于配置文件使用LLM进行评估

对一个数据集设置LLM评判评估，你需要配置三个主要组件：

1. 数据集读取配置

```python
reader_cfg = dict(
    input_columns=['problem'],  # 问题列的名称
    output_column='answer'      # 参考答案列的名称
)
```

2. 推理配置

```python
infer_cfg = dict(
    prompt_template=dict(
        type=PromptTemplate,
        template=dict(
            round=[
                dict(
                    role='HUMAN',
                    prompt='{problem}',  # 提示模型的模板
                ),
            ]
        ),
    ),
    retriever=dict(type=ZeroRetriever),
    inferencer=dict(type=GenInferencer),
)
```

3. 使用LLM评判器的评估配置

```python
eval_cfg = dict(
    evaluator=dict(
        type=GenericLLMEvaluator,  # 使用LLM作为评估器
        prompt_template=dict(
            type=PromptTemplate,
            template=dict(
                begin=[
                    dict(
                        role='SYSTEM',
                        fallback_role='HUMAN',
                        prompt="你是一个负责评估模型输出正确性和质量的助手。",
                    )
                ],
                round=[
                    dict(role='HUMAN', prompt=YOUR_JUDGE_TEMPLATE),  # 评判器的模板
                ],
            ),
        ),
        dataset_cfg=dict(
            type=CustomDataset,
            path='path/to/your/dataset',
            file_name='your_dataset.jsonl',
            reader_cfg=reader_cfg,
        ),
        judge_cfg=YOUR_JUDGE_MODEL_CONFIG,  # 评判模型的配置
        dict_postprocessor=dict(type=generic_llmjudge_postprocess),  # 处理评判器输出的后处理器
    ),
)
```

## 使用CustomDataset和GenericLLMEvaluator

以下是如何设置完整的LLM评判评估配置：

```python
from mmengine.config import read_base
from opencompass.models import TurboMindModelwithChatTemplate
from opencompass.datasets import CustomDataset
from opencompass.evaluator import GenericLLMEvaluator
from opencompass.datasets import generic_llmjudge_postprocess
from opencompass.openicl.icl_prompt_template import PromptTemplate
from opencompass.openicl.icl_retriever import ZeroRetriever
from opencompass.openicl.icl_inferencer import GenInferencer

# 导入评判模型配置
with read_base():
    from opencompass.configs.models.qwen2_5.lmdeploy_qwen2_5_14b_instruct import (
        models as judge_model,
    )

# 定义评判模板
JUDGE_TEMPLATE = """
请评估以下回答是否正确地回答了问题。
问题：{problem}
参考答案：{answer}
模型回答：{prediction}

模型回答是否正确？如果正确，请回答"A"；如果不正确，请回答"B"。
""".strip()

# 数据集读取配置
reader_cfg = dict(input_columns=['problem'], output_column='answer')

# 被评估模型的推理配置
infer_cfg = dict(
    prompt_template=dict(
        type=PromptTemplate,
        template=dict(
            round=[
                dict(
                    role='HUMAN',
                    prompt='{problem}',
                ),
            ]
        ),
    ),
    retriever=dict(type=ZeroRetriever),
    inferencer=dict(type=GenInferencer),
)

# 使用LLM评判器的评估配置
eval_cfg = dict(
    evaluator=dict(
        type=GenericLLMEvaluator,
        prompt_template=dict(
            type=PromptTemplate,
            template=dict(
                begin=[
                    dict(
                        role='SYSTEM',
                        fallback_role='HUMAN',
                        prompt="你是一个负责评估模型输出正确性和质量的助手。",
                    )
                ],
                round=[
                    dict(role='HUMAN', prompt=JUDGE_TEMPLATE),
                ],
            ),
        ),
        dataset_cfg=dict(
            type=CustomDataset,
            path='path/to/your/dataset',
            file_name='your_dataset.jsonl',
            reader_cfg=reader_cfg,
        ),
        judge_cfg=judge_model[0],
        dict_postprocessor=dict(type=generic_llmjudge_postprocess),
    ),
    pred_role='BOT',
)

# 数据集配置
datasets = [
    dict(
        type=CustomDataset,
        abbr='my-dataset',
        path='path/to/your/dataset',
        file_name='your_dataset.jsonl',
        reader_cfg=reader_cfg,
        infer_cfg=infer_cfg,
        eval_cfg=eval_cfg,
    )
]

# 被评估模型的配置
models = [
    dict(
        type=TurboMindModelwithChatTemplate,
        abbr='model-to-evaluate',
        path='path/to/your/model',
        # ... 其他模型配置
    )
]

# 输出目录
work_dir = './outputs/llm_judge_eval'
```

## GenericLLMEvaluator

GenericLLMEvaluator专为使用LLM作为评判器评估模型输出而设计。主要特点包括：

1. 灵活的提示模板，用于指导评判器
2. 支持各种评判模型（本地或基于API）
3. 通过提示工程自定义评估标准
4. 对评判器输出进行后处理以提取结构化评估

**重要说明**：目前通用版本的评判模板只支持输出"A"（正确）或"B"（不正确）的格式，不支持其他输出格式（如"正确"或"不正确"）。这是因为后处理函数`generic_llmjudge_postprocess`专门设计为解析这种格式。

评估器的工作原理：

1. 获取原始问题、参考答案和模型预测
2. 将它们格式化为评判模型的提示
3. 解析评判器的响应以确定评估结果（寻找"A"或"B"）
4. 汇总整个数据集的结果

如果需要查看评估的详细结果，可以在启动任务时添加`--dump-eval-details`到命令行。
评估输出示例：

```python
{
    'accuracy': 75.0,  # 被判断为正确的回答百分比
    'details': [
        {
            'origin_prompt': """
            请评估以下回答是否正确地回答了问题。
            问题：法国的首都是什么？
            参考答案：巴黎
            模型回答：法国的首都是巴黎。
            模型回答是否正确？如果正确，请回答"A"；如果不正确，请回答"B"。""",
            'gold': '巴黎',
            'prediction': 'A',
        },
        # ... 更多结果
    ]
}
```

## 级联评估器 (CascadeEvaluator)

OpenCompass还提供了级联评估器`CascadeEvaluator`，它结合了规则式评估和LLM评估的优势。级联评估器有两种模式：

1. **级联模式（Cascade Mode, parallel=False）**：首先使用规则式评估器评估所有样本，然后只将规则式评估认为不正确的样本发送给LLM评判器进行重新评估。这种方式可以在保持准确性的同时减少对LLM评判的依赖，从而降低评估成本和时间。

2. **并行模式（Parallel Mode, parallel=True）**：使用规则式评估器和LLM评判器同时评估所有样本，如果任何一个评估器认为样本是正确的，则将该样本视为正确。这种方式可以提高评估的宽容度，但可能会导致更高的成本，因为所有样本都需要LLM评估。

### 配置CascadeEvaluator

以下是配置`CascadeEvaluator`的示例：

```python
# 定义规则式评估器
rule_evaluator = dict(type=MATHEvaluator)

# 定义LLM评判器
llm_judge_evaluator = dict(
    type=GenericLLMEvaluator,
    prompt_template=dict(
        type=PromptTemplate,
        template=dict(
            begin=[
                dict(
                    role='SYSTEM',
                    fallback_role='HUMAN',
                    prompt="你是一个负责评估模型输出正确性和质量的助手。",
                )
            ],
            round=[
                dict(role='HUMAN', prompt=YOUR_JUDGE_TEMPLATE),
            ],
        ),
    ),
    dataset_cfg=dict(
        type=YourDataset,
        path='path/to/your/dataset',
        reader_cfg=reader_cfg,
    ),
    judge_cfg=dict(),  # 可以使用环境变量配置评判模型
)

# 配置级联评估器（级联模式）
cascade_evaluator = dict(
    type=CascadeEvaluator,
    llm_evaluator=llm_judge_evaluator,
    rule_evaluator=rule_evaluator,
    parallel=False  # 级联模式
)

# 如果需要并行模式，可以设置parallel=True
parallel_evaluator = dict(
    type=CascadeEvaluator,
    llm_evaluator=llm_judge_evaluator,
    rule_evaluator=rule_evaluator,
    parallel=True  # 并行模式
)

# 在数据集评估配置中使用级联评估器
eval_cfg = dict(evaluator=cascade_evaluator)
```

### 评估结果

级联评估器会输出详细的评估统计信息，包括：

- 规则评估的准确率
- LLM评估的准确率（针对规则评估失败的样本）
- 最终的综合准确率

输出示例：

```python
{
    'accuracy': 85.0,  # 最终准确率
    'cascade_stats': {
        'total_samples': 100,
        'rule_correct': 70,  # 规则评估认为正确的样本数
        'rule_accuracy': 70.0,  # 规则评估的准确率
        'llm_evaluated': 30,  # LLM评估的样本数（级联模式下为规则评估失败的样本数）
        'llm_correct': 15,  # LLM评估认为正确的样本数
        'llm_accuracy': 50.0,  # LLM评估的准确率
        'final_correct': 85,  # 最终正确的样本数
        'final_accuracy': 85.0,  # 最终准确率
        'parallel_mode': False,  # 是否是并行模式
    },
    'details': [
        # 每个样本的详细评估结果
    ]
}
```

级联评估器特别适用于：

1. 需要平衡评估成本和准确性的场景
2. 有可用的规则式评估器但可能不够完善的情况
3. 需要对边界情况进行更精确判断的评估任务

## 完整示例

如果希望了解通用LLM评判器，请参考examples目录中的`eval_llm_judge.py`文件，该示例展示了如何使用LLM评判器评估数学问题。

如果希望了解级联评估器请参考examples目录中的`eval_cascade_evaluator.py`文件，该示例展示了如何使用级联评估器评估数学问题。
