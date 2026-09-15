# Mathematical Capability Evaluation

## Introduction

Mathematical reasoning is a crucial capability for large language models (LLMs). To evaluate a model's mathematical abilities, we need to test its capability to solve mathematical problems step by step and provide accurate final answers. OpenCompass provides a convenient way to evaluate mathematical reasoning through the CustomDataset and MATHVerifyEvaluator components.

## MATHVerifyEvaluator

MATHVerifyEvaluator is specifically designed to evaluate mathematical answers. It is developed based on the math_verify library, which provides mathematical expression parsing and verification capabilities, supporting extraction and equivalence verification for both LaTeX and general expressions.

MATHVerifyEvaluator implements:

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
            'pred': 'x = 2',     # Parsed prediction
            'answer': 'x = 2',   # Parsed reference
            'correct': True      # Whether they match
        },
        # ... more results
    ]
}
```

## MATHVerifyEvaluator Configuration

OpenCompass already includes configs that use MATHVerifyEvaluator, such as [opencompass/configs/datasets/math/math_500_gen.py](https://github.com/open-compass/opencompass/blob/main/opencompass/configs/datasets/math/math_500_gen.py#L1-L40). The following sections explain the relevant parts of that config.

### 1. Imports

The [config file](https://github.com/open-compass/opencompass/blob/main/opencompass/configs/datasets/math/math_500_gen.py#L1-L5) imports the dataset, inference, and evaluator components:

```python
from opencompass.openicl.icl_prompt_template import PromptTemplate
from opencompass.openicl.icl_retriever import ZeroRetriever
from opencompass.openicl.icl_inferencer import GenInferencer
from opencompass.datasets import CustomDataset
from opencompass.evaluator import MATHVerifyEvaluator
```

### 2. Dataset Reader Configuration

[reader_cfg](https://github.com/open-compass/opencompass/blob/main/opencompass/configs/datasets/math/math_500_gen.py#L7) tells OpenCompass which columns are used in the model prompt and which column is used as the reference answer:

```python
math_reader_cfg = dict(input_columns=['problem'], output_column='solution')
```

Here, `problem` is used by the `{problem}` placeholder in the inference prompt, and `solution` is used as the reference answer during evaluation.

### 3. Inference Configuration

[infer_cfg](https://github.com/open-compass/opencompass/blob/main/opencompass/configs/datasets/math/math_500_gen.py#L9-L23) defines the prompt seen by the model, the retriever, and the inferencer:

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

The prompt asks the model to reason step by step and place the final answer inside `\boxed{}`. MATHVerifyEvaluator extracts mathematical expressions from the prediction and reference answer, then checks whether they are equivalent.

### 4. MATHVerifyEvaluator Configuration Parameters

In [eval_cfg](https://github.com/open-compass/opencompass/blob/main/opencompass/configs/datasets/math/math_500_gen.py#L26-L28), only the evaluator type is required:

```python
math_eval_cfg = dict(
    evaluator=dict(type=MATHVerifyEvaluator),
)
```

The supported parameters when configuring MATHVerifyEvaluator are:

- `type`: required. Specifies `MATHVerifyEvaluator` as the evaluator.
- `pred_postprocessor`: optional. This generic parameter is inherited from `BaseEvaluator`. When configured inside `evaluator`, it post-processes model predictions before mathematical answer verification, which is useful for removing fixed prefixes or extracting a specific answer span.

For example:

```python
math_eval_cfg = dict(
    evaluator=dict(
        type=MATHVerifyEvaluator,
        # pred_postprocessor=dict(type=your_postprocess),  # optional
    ),
)
```

MATHVerifyEvaluator currently does not expose additional evaluator-specific config options. Mathematical expression extraction, equivalence verification, and the per-sample 10-second timeout are fixed by the evaluator implementation and cannot be changed directly through extra config parameters.

OpenICL tasks also support placing `pred_postprocessor` at the top level of `eval_cfg`:

```python
math_eval_cfg = dict(
    evaluator=dict(type=MATHVerifyEvaluator),
    pred_postprocessor=dict(type=your_postprocess),
)
```

This is generic dataset evaluation post-processing and runs before the evaluator is built and called. If `pred_postprocessor` is configured both at the top level of `eval_cfg` and inside `evaluator`, predictions will be processed twice, which is usually not recommended.

### 5. Dataset Configuration

Finally, [math_datasets](https://github.com/open-compass/opencompass/blob/main/opencompass/configs/datasets/math/math_500_gen.py#L30-L40) combines the reader, inference, and evaluation configs:

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

## Using CustomDataset

To evaluate your own math dataset, you can load it with the built-in `CustomDataset` and combine it with MATHVerifyEvaluator. `CustomDataset` currently supports `.jsonl` and `.csv`; each problem usually contains at least:

- A problem statement, such as `problem`
- A solution/answer, such as `solution` (typically in LaTeX format with the final answer in `\boxed{}`)

Example JSONL format:

```json
{"problem": "Find the value of x if 2x + 3 = 7", "solution": "Let's solve step by step:\n2x + 3 = 7\n2x = 7 - 3\n2x = 4\nx = 2\nTherefore, \\boxed{2}"}
```

Example CSV format:

```text
problem,solution
"Find the value of x if 2x + 3 = 7","Let's solve step by step:\n2x + 3 = 7\n2x = 7 - 3\n2x = 4\nx = 2\nTherefore, \\boxed{2}"
```

### 1. Dataset Reader Configuration

```python
math_reader_cfg = dict(
    input_columns=['problem'],
    output_column='solution',
)
```

`input_columns` specifies the fields used by the inference template, so the prompt can use `{problem}`. `output_column` specifies the reference answer field used during evaluation; here, `solution` is passed to MATHVerifyEvaluator as the reference.

### 2. Inference Configuration

```python
math_infer_cfg = dict(
    prompt_template=dict(
        type=RawPromptTemplate,
        messages=[
            dict(
                role='user',
                content='{problem}\nPlease reason step by step, and put your final answer within \\boxed{}.',
            ),
        ],
    ),
    retriever=dict(type=ZeroRetriever),
    inferencer=dict(type=GenInferencer),
)
```

`RawPromptTemplate` describes chat messages directly through `messages`. The `{problem}` placeholder comes from `reader_cfg.input_columns` and is replaced with the problem text from each sample. The prompt asks the model to place the final answer inside `\boxed{}`, which helps later extraction and equivalence verification.

### 3. Evaluation Configuration

```python
math_eval_cfg = dict(
    evaluator=dict(type=MATHVerifyEvaluator),
)
```

`eval_cfg` uses MATHVerifyEvaluator to verify mathematical equivalence between predictions and references. If the model output needs cleanup before verification, add `pred_postprocessor` as described above.

### 4. Dataset Configuration

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

`path` and `file_name` point to the data file to evaluate. `reader_cfg`, `infer_cfg`, and `eval_cfg` connect the reading, inference, and evaluation stages.

### Complete Config

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
                content='{problem}\nPlease reason step by step, and put your final answer within \\boxed{}.',
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
