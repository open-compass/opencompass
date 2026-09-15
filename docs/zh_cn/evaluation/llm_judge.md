# LLM 作为评判器

## 简介

GenericLLMEvaluator组件特别适用于那些难以通过规则式方法（如正则表达式）进行完美判断的场景，例如：

- 模型不输出选项标识而只输出选项内容的情况
- 需要事实性判断的数据集
- 需要复杂理解和推理的开放式回答
- 需要设计大量规则的判断

OpenCompass提供了GenericLLMEvaluator组件来实现LLM作为评判器的评估。

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

注意，默认情况下，OpenCompass会使用这三个环境变量；但如果在配置文件中显式提供了具体的 `judge_cfg`，这三个环境变量将不会生效。

### 基于配置文件使用LLM进行评估

可以参考 HLE 的现有配置：[opencompass/configs/datasets/HLE/hle_llmverify_rawprompt_gen_0970dd.py](https://github.com/open-compass/opencompass/blob/main/opencompass/configs/datasets/HLE/hle_llmverify_rawprompt_gen_0970dd.py#L1-L80)。这份配置使用 `HLEDataset`、`RawPromptTemplate` 和 `GenericLLMEvaluator`，并将 `judge_cfg` 留空，以便从 `OC_JUDGE_MODEL` / `OC_JUDGE_API_KEY` / `OC_JUDGE_API_BASE` 读取评判模型配置。

代码中用到的组件来自这些 import（[第 1-6 行](https://github.com/open-compass/opencompass/blob/main/opencompass/configs/datasets/HLE/hle_llmverify_rawprompt_gen_0970dd.py#L1-L6)）：

```python
from opencompass.openicl.icl_raw_prompt_template import RawPromptTemplate
from opencompass.openicl.icl_retriever import ZeroRetriever
from opencompass.openicl.icl_inferencer import GenInferencer
from opencompass.evaluator import GenericLLMEvaluator
from opencompass.datasets import generic_llmjudge_postprocess
from opencompass.datasets import HLEDataset
```

#### 1. 数据集读取配置

`reader_cfg` 定义哪些字段进入被评测模型的输入，以及哪一列作为参考答案。HLE 使用 `problem` 作为问题列，`answer` 作为参考答案列（[第 10 行](https://github.com/open-compass/opencompass/blob/main/opencompass/configs/datasets/HLE/hle_llmverify_rawprompt_gen_0970dd.py#L10)）：

```python
math_reader_cfg = dict(input_columns=['problem'], output_column='answer')
```

#### 2. 推理配置

`infer_cfg` 定义被评测模型如何接收数据集字段。这里使用 `RawPromptTemplate` 将 `{problem}` 填入 user 消息，并要求模型把最终答案放在 `\boxed{}` 中（[第 12-21 行](https://github.com/open-compass/opencompass/blob/main/opencompass/configs/datasets/HLE/hle_llmverify_rawprompt_gen_0970dd.py#L12-L21)）：

```python
math_infer_cfg = dict(
    prompt_template=dict(
        type=RawPromptTemplate,
        messages=[
            {'role': 'user', 'content': '{problem}\nRemember to put your final answer within \\boxed{}.'},
        ]
    ),
    retriever=dict(type=ZeroRetriever),
    inferencer=dict(type=GenInferencer),
)
```

#### 3. 使用 LLM 评判器的评估配置

`GRADER_TEMPLATE` 是给评判模型的提示模板，它会引用原始问题 `{problem}`、参考答案 `{answer}` 和评测过程中追加的模型预测 `{prediction}`（[第 23-46 行](https://github.com/open-compass/opencompass/blob/main/opencompass/configs/datasets/HLE/hle_llmverify_rawprompt_gen_0970dd.py#L23-L46)）：

`GenericLLMEvaluator` 的评判 prompt 没有固定的占位符白名单。使用 `RawPromptTemplate` 时，`messages[*].content` 中形如 `{column_name}` 的内容会用评判阶段当前样本里的同名字段替换；字段不存在时不会被替换，会以原文本形式保留在 prompt 中。

可用字段通常包括两类：

- 原始评测数据集中的列，例如 HLE 的 `problem`、`answer`。如果模板写 `{answer}`，评判用的 `dataset_cfg` 加载出的样本里就需要存在 `answer` 这一列。
- `GenericLLMEvaluator` 在评判前追加的列：`prediction` 是被评测模型的输出；`reference` 是传入 evaluator 的参考答案，通常来自 `reader_cfg.output_column` 并已应用 `dataset_postprocessor`；`obj_gold` 与 `reference` 使用同一批参考答案。

因此，HLE 配置里的三个占位符来源如下：

- `{problem}` 来自原始 HLE 数据集中的 `problem` 列。该列同时在 `math_reader_cfg.input_columns` 中声明，因此也会作为被评测模型的输入字段。
- `{answer}` 来自原始 HLE 数据集中的 `answer` 列，也就是 `math_reader_cfg.output_column` 指定的参考答案列。
- `{prediction}` 不是原始数据集字段，而是被评测模型完成推理后，由 `GenericLLMEvaluator` 在评判前追加到评判数据中的模型输出字段。

```python
GRADER_TEMPLATE = """
    Please as a grading expert, judge whether the final answers given by the candidates below are consistent with the standard answers, that is, whether the candidates answered correctly.

    Here are some evaluation criteria:
    1. Please refer to the given standard answer. You don't need to re-generate the answer to the question because the standard answer has been given. You only need to judge whether the candidate's answer is consistent with the standard answer according to the form of the question. Don't try to answer the original question. You can assume that the standard answer is definitely correct.
    2. Because the candidate's answer may be different from the standard answer in the form of expression, before making a judgment, please understand the question and the standard answer first, and then judge whether the candidate's answer is correct, but be careful not to try to answer the original question.
    3. Some answers may contain multiple items, such as multiple-choice questions, multiple-select questions, fill-in-the-blank questions, etc. As long as the answer is the same as the standard answer, it is enough. For multiple-select questions and multiple-blank fill-in-the-blank questions, the candidate needs to answer all the corresponding options or blanks correctly to be considered correct.
    4. Some answers may be expressed in different ways, such as some answers may be a mathematical expression, some answers may be a textual description, as long as the meaning expressed is the same. And some formulas are expressed in different ways, but they are equivalent and correct.
    5. If the prediction is given with \\boxed{}, please ignore the \\boxed{} and only judge whether the candidate's answer is consistent with the standard answer.

    Please judge whether the following answers are consistent with the standard answer based on the above criteria. Grade the predicted answer of this new question as one of:
    A: CORRECT
    B: INCORRECT
    Just return the letters "A" or "B", with no text around it.

    Here is your task. Simply reply with either CORRECT, INCORRECT. Don't apologize or correct yourself if there was a mistake; we are just trying to grade the answer.


    <Original Question Begin>: \n{problem}\n<Original Question End>\n\n
    <Gold Target Begin>: \n{answer}\n<Gold Target End>\n\n
    <Predicted Answer Begin>: \n{prediction}\n<Predicted End>\n\n

    Judging the correctness of candidates' answers:
""".strip()
```

`math_eval_cfg` 使用 `GenericLLMEvaluator` 发起评判模型调用，并通过 `generic_llmjudge_postprocess` 将评判输出解析为可汇总的结果。`judge_cfg=dict()` 表示这里不在配置文件中写死评判模型，而是从环境变量读取（[第 48-68 行](https://github.com/open-compass/opencompass/blob/main/opencompass/configs/datasets/HLE/hle_llmverify_rawprompt_gen_0970dd.py#L48-L68)）：

```python
math_eval_cfg = dict(
    evaluator=dict(
        type=GenericLLMEvaluator,
        prompt_template=dict(
            type=RawPromptTemplate,
            messages=[
                {'role': 'system', 'content': "You are a helpful assistant who evaluates the correctness and quality of models' outputs."},
                {'role': 'user', 'content': GRADER_TEMPLATE},
            ],
        ),
        dataset_cfg=dict(
            type=HLEDataset,
            path='cais/hle',
            reader_cfg=math_reader_cfg,
        ),
        judge_cfg=dict(),
        dict_postprocessor=dict(type=generic_llmjudge_postprocess),
    ),
    pred_role='BOT',
)
```

#### 4. 数据集配置

最后在数据集配置中把 `reader_cfg`、`infer_cfg` 和 `eval_cfg` 组合起来，形成可被 OpenCompass 加载的 `hle_datasets`（[第 71-80 行](https://github.com/open-compass/opencompass/blob/main/opencompass/configs/datasets/HLE/hle_llmverify_rawprompt_gen_0970dd.py#L71-L80)）：

```python
hle_datasets = [
    dict(
        type=HLEDataset,
        abbr='hle_llmjudge',
        path='cais/hle',
        reader_cfg=math_reader_cfg,
        infer_cfg=math_infer_cfg,
        eval_cfg=math_eval_cfg,
    )
]
```

## 使用CustomDataset和GenericLLMEvaluator

### 数据集与列配置

本节只说明使用内置 `CustomDataset` 时的数据文件和列配置。评判 prompt 可引用哪些字段，见上文 `GenericLLMEvaluator` 的占位符说明。

`CustomDataset` 当前支持 `.jsonl` 和 `.csv`。数据文件中至少应包含 judge 模板需要引用的原始字段，例如：

- 问题或任务
- 参考答案或标准答案

参考答案列名需要与 `reader_cfg.output_column` 保持一致，例如 `answer`、`gold` 或 `target`。模型预测不需要写在原始数据文件中，会在评测过程中生成并提供给评判器。

JSONL 格式示例：

```json
{"problem": "法国的首都是什么？", "answer": "巴黎"}
```

CSV 格式示例：

```text
problem,answer
"法国的首都是什么？","巴黎"
```

以下是如何设置完整的LLM评判评估配置：

```python
from mmengine.config import read_base
from opencompass.models import TurboMindModelwithChatTemplate
from opencompass.datasets import CustomDataset
from opencompass.evaluator import GenericLLMEvaluator
from opencompass.datasets import generic_llmjudge_postprocess
from opencompass.openicl.icl_raw_prompt_template import RawPromptTemplate
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
        type=RawPromptTemplate,
        messages=[
            dict(role='user', content='{problem}'),
        ],
    ),
    retriever=dict(type=ZeroRetriever),
    inferencer=dict(type=GenInferencer),
)

# 使用LLM评判器的评估配置
eval_cfg = dict(
    evaluator=dict(
        type=GenericLLMEvaluator,
        prompt_template=dict(
            type=RawPromptTemplate,
            messages=[
                dict(role='system', content='你是一个负责评估模型输出正确性和质量的助手。'),
                dict(role='user', content=JUDGE_TEMPLATE),
            ],
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

## 示例脚本

如果希望查看一份可作为配置运行的通用 LLM 评判器示例，请参考 `examples/eval_llm_judge.py`。该脚本使用 `CustomDataset` 和 `GenericLLMEvaluator` 演示了如何在数学问题上完成推理、调用评判模型并汇总结果。

## 级联评估器 (CascadeEvaluator)

规则式评估和 LLM 评估可以级联组合：规则先行，判错的样本再交给 LLM 评判复核。`CascadeEvaluator` 的两种模式、判定口径、缓存与复跑行为详见[级联评测](cascade_evaluator.md)。
