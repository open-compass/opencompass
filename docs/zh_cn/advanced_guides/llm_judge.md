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

要设置LLM评判评估，你需要配置三个主要组件：

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

## 完整示例

有关完整的工作示例，请参考examples目录中的`eval_llm_judge.py`文件，该文件演示了如何使用LLM评判器评估数学问题解决能力。
