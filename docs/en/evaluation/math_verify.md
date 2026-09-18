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

OpenCompass already includes configs that use MATHVerifyEvaluator directly, such as the [AIME 2026 MATHVerify config](https://github.com/open-compass/opencompass/blob/main/opencompass/configs/datasets/aime2026/aime2026_mathverify_rawprompt_gen_0970dd.py). The following sections explain its main parts.

### 1. Imports

The [config file](https://github.com/open-compass/opencompass/blob/main/opencompass/configs/datasets/aime2026/aime2026_mathverify_rawprompt_gen_0970dd.py#L1-L5) imports the dataset, inference, and evaluation components:

```python
from opencompass.datasets import CustomDataset
from opencompass.evaluator import MATHVerifyEvaluator
from opencompass.openicl.icl_inferencer import GenInferencer
from opencompass.openicl.icl_raw_prompt_template import RawPromptTemplate
from opencompass.openicl.icl_retriever import ZeroRetriever
```

### 2. Dataset Reader Configuration

[reader_cfg](https://github.com/open-compass/opencompass/blob/main/opencompass/configs/datasets/aime2026/aime2026_mathverify_rawprompt_gen_0970dd.py#L7) tells OpenCompass which columns are used in the model prompt and which column is used as the reference answer:

```python
aime2026_reader_cfg = dict(input_columns=['problem'], output_column='answer')
```

Here, `problem` is used by the `{problem}` placeholder in the inference prompt, and `answer` is used as the reference answer during evaluation.

### 3. Inference Configuration

[infer_cfg](https://github.com/open-compass/opencompass/blob/main/opencompass/configs/datasets/aime2026/aime2026_mathverify_rawprompt_gen_0970dd.py#L9-L21) defines the prompt seen by the model, the retriever, and the inferencer:

```python
aime2026_infer_cfg = dict(
    prompt_template=dict(
        type=RawPromptTemplate,
        messages=[
            {
                'role': 'user',
                'content': '{problem}\nRemember to put your final answer within \\boxed{}.',
            },
        ],
    ),
    retriever=dict(type=ZeroRetriever),
    inferencer=dict(type=GenInferencer),
)
```

This config uses `RawPromptTemplate` to construct the user message directly and asks the model to put its final answer inside `\boxed{}`, making it easier for MATHVerifyEvaluator to extract and verify the answer.

### 4. MATHVerifyEvaluator Configuration

[eval_cfg](https://github.com/open-compass/opencompass/blob/main/opencompass/configs/datasets/aime2026/aime2026_mathverify_rawprompt_gen_0970dd.py#L23-L25) selects MATHVerifyEvaluator directly:

```python
aime2026_eval_cfg = dict(
    evaluator=dict(type=MATHVerifyEvaluator),
)
```

During evaluation, MATHVerifyEvaluator extracts mathematical expressions from both the model predictions and the `answer` references, then checks whether they are equivalent. It currently exposes no evaluator-specific configuration options; mathematical-expression extraction, equivalence verification, and the per-sample 10-second timeout are fixed in its implementation and cannot be changed through additional config arguments.

### 5. Dataset Configuration

Finally, [aime2026_datasets](https://github.com/open-compass/opencompass/blob/main/opencompass/configs/datasets/aime2026/aime2026_mathverify_rawprompt_gen_0970dd.py#L27-L37) combines the reader, inference, and evaluation configs:

```python
aime2026_datasets = [
    dict(
        type=CustomDataset,
        abbr='aime2026',
        path='opencompass/aime2026',
        reader_cfg=aime2026_reader_cfg,
        infer_cfg=aime2026_infer_cfg,
        eval_cfg=aime2026_eval_cfg,
        n=1,
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
