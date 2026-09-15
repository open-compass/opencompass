# 数学能力评测

## 简介

数学推理能力是大语言模型(LLMs)的一项关键能力。为了评估模型的数学能力，我们需要测试其逐步解决数学问题并提供准确最终答案的能力。OpenCompass 通过 CustomDataset 和 MATHVerifyEvaluator 组件提供了一种便捷的数学推理评测方式。

## MATHVerifyEvaluator

MATHVerifyEvaluator 是专门设计用于评估数学答案的评测器。它基于 math_verify 库进行开发，该库提供了数学表达式解析和验证功能，支持 LaTeX 和一般表达式的提取与等价性验证。

MATHVerifyEvaluator 具有以下功能：

1. 使用 LaTeX 提取器从预测和参考答案中提取答案
2. 处理各种 LaTeX 格式和环境
3. 验证预测答案和参考答案之间的数学等价性
4. 提供详细的评测结果，包括：
   - 准确率分数
   - 预测和参考答案的详细比较
   - 预测和参考答案的解析结果

评测器支持：

- 基本算术运算
- 分数和小数
- 代数表达式
- 三角函数
- 根式和指数
- 数学符号和运算符

评测输出示例：

```python
{
    'accuracy': 85.0,  # 正确答案的百分比
    'details': [
        {
            'pred': 'x = 2',     # 解析后的预测答案
            'answer': 'x = 2',   # 解析后的参考答案
            'correct': True      # 是否匹配
        },
        # ... 更多结果
    ]
}
```

## MATHVerifyEvaluator 配置说明

OpenCompass 中已有使用 MATHVerifyEvaluator 的配置，例如 [opencompass/configs/datasets/math/math_500_gen.py](https://github.com/open-compass/opencompass/blob/main/opencompass/configs/datasets/math/math_500_gen.py#L1-L40)。下面按配置文件中的几个部分说明如何配置。

### 1. 导入依赖

[配置文件](https://github.com/open-compass/opencompass/blob/main/opencompass/configs/datasets/math/math_500_gen.py#L1-L5) 中导入了数据集、推理和评测所需组件：

```python
from opencompass.openicl.icl_prompt_template import PromptTemplate
from opencompass.openicl.icl_retriever import ZeroRetriever
from opencompass.openicl.icl_inferencer import GenInferencer
from opencompass.datasets import CustomDataset
from opencompass.evaluator import MATHVerifyEvaluator
```

### 2. 数据集读取配置

[reader_cfg](https://github.com/open-compass/opencompass/blob/main/opencompass/configs/datasets/math/math_500_gen.py#L7) 负责告诉 OpenCompass 哪些列会进入模型提示词、哪一列作为参考答案：

```python
math_reader_cfg = dict(input_columns=['problem'], output_column='solution')
```

其中 `problem` 会被用于推理模板中的 `{problem}` 占位符，`solution` 会作为评测阶段的参考答案。

### 3. 推理配置

[infer_cfg](https://github.com/open-compass/opencompass/blob/main/opencompass/configs/datasets/math/math_500_gen.py#L9-L23) 定义模型看到的提示词、检索器和推理器：

```python
math_infer_cfg = dict(
    prompt_template=dict(
        type=PromptTemplate,
        template=dict(
            round=[
                dict(
                    role='HUMAN',
                    prompt='{problem}\nPlease reason step by step, and put your final answer within \\boxed{}.',
                ),
            ]
        ),
    ),
    retriever=dict(type=ZeroRetriever),
    inferencer=dict(type=GenInferencer),
)
```

这里的提示词要求模型逐步推理，并把最终答案放在 `\boxed{}` 中。MATHVerifyEvaluator 会从预测和参考答案中抽取数学表达式，再判断二者是否等价。

### 4. MATHVerifyEvaluator 配置参数

[eval_cfg](https://github.com/open-compass/opencompass/blob/main/opencompass/configs/datasets/math/math_500_gen.py#L26-L28) 中只需要指定评测器类型：

```python
math_eval_cfg = dict(
    evaluator=dict(type=MATHVerifyEvaluator),
)
```

配置 MATHVerifyEvaluator 时支持的参数主要有：

- `type`：必填，指定使用 `MATHVerifyEvaluator`。
- `pred_postprocessor`：可选，继承自 `BaseEvaluator` 的通用参数。配置在 `evaluator` 内部时，会在数学答案验证前对模型预测结果做文本后处理，适合去掉固定前缀、截取特定答案片段等。

例如：

```python
math_eval_cfg = dict(
    evaluator=dict(
        type=MATHVerifyEvaluator,
        # pred_postprocessor=dict(type=your_postprocess),  # 可选
    ),
)
```

当前 MATHVerifyEvaluator 没有额外暴露专用配置项。数学表达式抽取、等价性验证和单样本 10 秒超时逻辑由评测器实现固定控制，不能直接在配置文件中通过额外参数修改。

另外，OpenICL 任务也支持把 `pred_postprocessor` 配在 `eval_cfg` 顶层：

```python
math_eval_cfg = dict(
    evaluator=dict(type=MATHVerifyEvaluator),
    pred_postprocessor=dict(type=your_postprocess),
)
```

这属于数据集评测配置的通用后处理，会在构造并调用 evaluator 之前执行；如果同时在 `eval_cfg` 顶层和 `evaluator` 内部配置 `pred_postprocessor`，预测结果会被处理两次，一般不建议这样做。

### 5. 数据集配置

最后在 [math_datasets](https://github.com/open-compass/opencompass/blob/main/opencompass/configs/datasets/math/math_500_gen.py#L30-L40) 中把读取、推理和评测配置组合起来：

```python
math_datasets = [
    dict(
        type=CustomDataset,
        abbr='math-500',
        path='opencompass/math',
        file_name='test_prm800k_500.jsonl',
        reader_cfg=math_reader_cfg,
        infer_cfg=math_infer_cfg,
        eval_cfg=math_eval_cfg,
    )
]
```

## 使用 CustomDataset

如果要评测自己的数学数据集，可以使用内置 `CustomDataset` 加载数据，并把它和 MATHVerifyEvaluator 组合起来。`CustomDataset` 当前支持 `.jsonl` 和 `.csv`；每个问题通常至少应包含：

- 问题陈述，例如 `problem`
- 解答/答案，例如 `solution`（通常使用 LaTeX 格式，最终答案建议用 `\boxed{}` 括起来）

JSONL 格式示例：

```json
{"problem": "求解方程 2x + 3 = 7", "solution": "让我们逐步解决：\n2x + 3 = 7\n2x = 7 - 3\n2x = 4\nx = 2\n因此，\\boxed{2}"}
```

CSV 格式示例：

```text
problem,solution
"求解方程 2x + 3 = 7","让我们逐步解决：\n2x + 3 = 7\n2x = 7 - 3\n2x = 4\nx = 2\n因此，\\boxed{2}"
```

### 1. 数据集读取配置

```python
math_reader_cfg = dict(
    input_columns=['problem'],
    output_column='solution',
)
```

`input_columns` 指定进入推理模板的字段，所以上面的推理模板可以使用 `{problem}`。`output_column` 指定评测时使用的参考答案字段，这里会把 `solution` 作为 MATHVerifyEvaluator 的 reference。

### 2. 推理配置

```python
math_infer_cfg = dict(
    prompt_template=dict(
        type=RawPromptTemplate,
        messages=[
            dict(
                role='user',
                content='{problem}\n请逐步推理，并将最终答案放在 \\boxed{} 中。',
            ),
        ],
    ),
    retriever=dict(type=ZeroRetriever),
    inferencer=dict(type=GenInferencer),
)
```

`RawPromptTemplate` 使用 `messages` 直接描述对话消息。这里的 `{problem}` 来自 `reader_cfg.input_columns`，推理时会被替换为样本中的题目内容；提示词要求模型把最终答案放进 `\boxed{}`，便于后续抽取和等价性验证。

### 3. 评测配置

```python
math_eval_cfg = dict(
    evaluator=dict(type=MATHVerifyEvaluator),
)
```

`eval_cfg` 指定使用 MATHVerifyEvaluator 对预测和参考答案做数学等价性验证。如果需要先清理模型输出，可以按前文说明添加 `pred_postprocessor`。

### 4. 数据集配置

```python
math_datasets = [
    dict(
        type=CustomDataset,
        abbr='my-math-dataset',
        path='path/to/your/dataset',
        file_name='your_dataset.jsonl',
        reader_cfg=math_reader_cfg,
        infer_cfg=math_infer_cfg,
        eval_cfg=math_eval_cfg,
    )
]
```

`path` 和 `file_name` 指向待评测数据文件，`reader_cfg`、`infer_cfg` 和 `eval_cfg` 分别接入读取、推理和评测流程。

### 完整配置

```python
from opencompass.openicl.icl_prompt_template import RawPromptTemplate
from opencompass.openicl.icl_retriever import ZeroRetriever
from opencompass.openicl.icl_inferencer import GenInferencer
from opencompass.datasets import CustomDataset
from opencompass.evaluator import MATHVerifyEvaluator

math_reader_cfg = dict(
    input_columns=['problem'],
    output_column='solution',
)

math_infer_cfg = dict(
    prompt_template=dict(
        type=RawPromptTemplate,
        messages=[
            dict(
                role='user',
                content='{problem}\n请逐步推理，并将最终答案放在 \\boxed{} 中。',
            ),
        ],
    ),
    retriever=dict(type=ZeroRetriever),
    inferencer=dict(type=GenInferencer),
)

math_eval_cfg = dict(
    evaluator=dict(type=MATHVerifyEvaluator),
)

math_datasets = [
    dict(
        type=CustomDataset,
        abbr='my-math-dataset',
        path='path/to/your/dataset',
        file_name='your_dataset.jsonl',
        reader_cfg=math_reader_cfg,
        infer_cfg=math_infer_cfg,
        eval_cfg=math_eval_cfg,
    )
]
```
