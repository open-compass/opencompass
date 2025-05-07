# General Math Evaluation Guidance

## Introduction

Mathematical reasoning is a crucial capability for large language models (LLMs). To evaluate a model's mathematical abilities, we need to test its capability to solve mathematical problems step by step and provide accurate final answers. OpenCompass provides a convenient way to evaluate mathematical reasoning through the CustomDataset and MATHVerifyEvaluator components.

## Dataset Format

The math evaluation dataset should be in either JSON Lines (.jsonl) or CSV format. Each problem should contain at least:

- A problem statement
- A solution/answer (typically in LaTeX format with the final answer in \\boxed{})

Example JSONL format:

```json
{"problem": "Find the value of x if 2x + 3 = 7", "solution": "Let's solve step by step:\n2x + 3 = 7\n2x = 7 - 3\n2x = 4\nx = 2\nTherefore, \\boxed{2}"}
```

Example CSV format:

```csv
problem,solution
"Find the value of x if 2x + 3 = 7","Let's solve step by step:\n2x + 3 = 7\n2x = 7 - 3\n2x = 4\nx = 2\nTherefore, \\boxed{2}"
```

## Configuration

To evaluate mathematical reasoning, you'll need to set up three main components:

1. Dataset Reader Configuration

```python
math_reader_cfg = dict(
    input_columns=['problem'],  # Column name for the question
    output_column='solution'    # Column name for the answer
)
```

2. Inference Configuration

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

3. Evaluation Configuration

```python
math_eval_cfg = dict(
    evaluator=dict(type=MATHVerifyEvaluator),
)
```

## Using CustomDataset

Here's how to set up a complete configuration for math evaluation:

```python
from mmengine.config import read_base
from opencompass.models import TurboMindModelwithChatTemplate
from opencompass.datasets import CustomDataset

math_datasets = [
    dict(
        type=CustomDataset,
        abbr='my-math-dataset',              # Dataset abbreviation
        path='path/to/your/dataset',         # Path to your dataset file
        reader_cfg=math_reader_cfg,
        infer_cfg=math_infer_cfg,
        eval_cfg=math_eval_cfg,
    )
]
```

## MATHVerifyEvaluator

The MATHVerifyEvaluator is specifically designed to evaluate mathematical answers. It is developed based on the math_verify library, which provides mathematical expression parsing and verification capabilities, supporting extraction and equivalence verification for both LaTeX and general expressions.

The MATHVerifyEvaluator implements:

1. Extracts answers from both predictions and references using LaTeX extraction
2. Handles various LaTeX formats and environments
3. Verifies mathematical equivalence between predicted and reference answers
4. Provides detailed evaluation results including:
   - Accuracy score
   - Detailed comparison between predictions and references
   - Parse results of both predicted and reference answers

The evaluator supports:

- Basic arithmetic operations
- Fractions and decimals
- Algebraic expressions
- Trigonometric functions
- Roots and exponents
- Mathematical symbols and operators

Example evaluation output:

```python
{
    'accuracy': 85.0,  # Percentage of correct answers
    'details': [
        {
            'predictions': 'x = 2',           # Parsed prediction
            'references': 'x = 2',         # Parsed reference
            'correct': True            # Whether they match
        },
        # ... more results
    ]
}
```

## Complete Example

Here's a complete example of how to set up math evaluation:

```python
from mmengine.config import read_base
from opencompass.models import TurboMindModelwithChatTemplate
from opencompass.datasets import CustomDataset
from opencompass.openicl.icl_evaluator.math_evaluator import MATHVerifyEvaluator
from opencompass.openicl.icl_prompt_template import PromptTemplate
from opencompass.openicl.icl_retriever import ZeroRetriever
from opencompass.openicl.icl_inferencer import GenInferencer

# Dataset reader configuration
math_reader_cfg = dict(input_columns=['problem'], output_column='solution')

# Inference configuration
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

# Evaluation configuration
math_eval_cfg = dict(
    evaluator=dict(type=MATHVerifyEvaluator),
)

# Dataset configuration
math_datasets = [
    dict(
        type=CustomDataset,
        abbr='my-math-dataset',
        path='path/to/your/dataset.jsonl',  # or .csv
        reader_cfg=math_reader_cfg,
        infer_cfg=math_infer_cfg,
        eval_cfg=math_eval_cfg,
    )
]

# Model configuration
models = [
    dict(
        type=TurboMindModelwithChatTemplate,
        abbr='your-model-name',
        path='your/model/path',
        # ... other model configurations
    )
]

# Output directory
work_dir = './outputs/math_eval'
```
