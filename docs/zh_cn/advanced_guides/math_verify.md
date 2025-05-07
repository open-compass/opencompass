# 数学能力评测

## 简介

数学推理能力是大语言模型(LLMs)的一项关键能力。为了评估模型的数学能力，我们需要测试其逐步解决数学问题并提供准确最终答案的能力。OpenCompass 通过 CustomDataset 和 MATHVerifyEvaluator 组件提供了一种便捷的数学推理评测方式。

## 数据集格式

数学评测数据集应该是 JSON Lines (.jsonl) 或 CSV 格式。每个问题至少应包含：

- 问题陈述
- 解答/答案（通常使用 LaTeX 格式，最终答案需要用 \\boxed{} 括起来）

JSONL 格式示例：

```json
{"problem": "求解方程 2x + 3 = 7", "solution": "让我们逐步解决：\n2x + 3 = 7\n2x = 7 - 3\n2x = 4\nx = 2\n因此，\\boxed{2}"}
```

CSV 格式示例：

```csv
problem,solution
"求解方程 2x + 3 = 7","让我们逐步解决：\n2x + 3 = 7\n2x = 7 - 3\n2x = 4\nx = 2\n因此，\\boxed{2}"
```

## 配置说明

要进行数学推理评测，你需要设置三个主要组件：

1. 数据集读取配置

```python
math_reader_cfg = dict(
    input_columns=['problem'],  # 问题列的名称
    output_column='solution'    # 答案列的名称
)
```

2. 推理配置

```python
math_infer_cfg = dict(
    prompt_template=dict(
        type=PromptTemplate,
        template=dict(
            round=[
                dict(
                    role='HUMAN',
                    prompt='{problem}\n请逐步推理，并将最终答案放在 \\boxed{} 中。',
                ),
            ]
        ),
    ),
    retriever=dict(type=ZeroRetriever),
    inferencer=dict(type=GenInferencer),
)
```

3. 评测配置

```python
math_eval_cfg = dict(
    evaluator=dict(type=MATHVerifyEvaluator),
)
```

## 使用 CustomDataset

以下是如何设置完整的数学评测配置：

```python
from mmengine.config import read_base
from opencompass.models import TurboMindModelwithChatTemplate
from opencompass.datasets import CustomDataset

math_datasets = [
    dict(
        type=CustomDataset,
        abbr='my-math-dataset',              # 数据集简称
        path='path/to/your/dataset',         # 数据集文件路径
        reader_cfg=math_reader_cfg,
        infer_cfg=math_infer_cfg,
        eval_cfg=math_eval_cfg,
    )
]
```

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
            'predictions': 'x = 2',           # 解析后的预测答案
            'references': 'x = 2',         # 解析后的参考答案
            'correct': True            # 是否匹配
        },
        # ... 更多结果
    ]
}
```

## 完整示例

以下是设置数学评测的完整示例：

```python
from mmengine.config import read_base
from opencompass.models import TurboMindModelwithChatTemplate
from opencompass.datasets import CustomDataset
from opencompass.evaluator import MATHVerifyEvaluator
from opencompass.openicl.icl_prompt_template import PromptTemplate
from opencompass.openicl.icl_retriever import ZeroRetriever
from opencompass.openicl.icl_inferencer import GenInferencer

# 数据集读取配置
math_reader_cfg = dict(input_columns=['problem'], output_column='solution')

# 推理配置
math_infer_cfg = dict(
    prompt_template=dict(
        type=PromptTemplate,
        template=dict(
            round=[
                dict(
                    role='HUMAN',
                    prompt='{problem}\n请逐步推理，并将最终答案放在 \\boxed{} 中。',
                ),
            ]
        ),
    ),
    retriever=dict(type=ZeroRetriever),
    inferencer=dict(type=GenInferencer),
)

# 评测配置
math_eval_cfg = dict(
    evaluator=dict(type=MATHVerifyEvaluator),
)

# 数据集配置
math_datasets = [
    dict(
        type=CustomDataset,
        abbr='my-math-dataset',
        path='path/to/your/dataset.jsonl',  # 或 .csv
        reader_cfg=math_reader_cfg,
        infer_cfg=math_infer_cfg,
        eval_cfg=math_eval_cfg,
    )
]

# 模型配置
models = [
    dict(
        type=TurboMindModelwithChatTemplate,
        abbr='your-model-name',
        path='your/model/path',
        # ... 其他模型配置
    )
]

# 输出目录
work_dir = './outputs/math_eval'
```
