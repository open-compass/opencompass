# LLM as Judge Evaluation

## Introduction

The GenericLLMEvaluator is particularly useful for scenarios where rule-based methods (like regular expressions) cannot perfectly judge outputs, such as:

- Cases where models output answer content without option identifiers
- Factual judgment datasets that are difficult to evaluate with rules
- Open-ended responses requiring complex understanding and reasoning
- Evaluation that requires a lot of rules to be designed

OpenCompass provides the GenericLLMEvaluator component to facilitate LLM-as-judge evaluations.

## Dataset Format

The dataset for LLM judge evaluation should be in either JSON Lines (.jsonl) or CSV format. Each entry should contain at least:

- A problem or question
- A reference answer or gold standard
- (The model's prediction will be generated during evaluation)

Example JSONL format:

```json
{"problem": "What is the capital of France?", "answer": "Paris"}
```

Example CSV format:

```csv
problem,answer
"What is the capital of France?","Paris"
```

## Configuration

To set up an LLM judge evaluation, you'll need to configure three main components:

1. Dataset Reader Configuration

```python
reader_cfg = dict(
    input_columns=['problem'],  # Column name for the question
    output_column='answer'      # Column name for the reference answer
)
```

2. Inference Configuration

```python
infer_cfg = dict(
    prompt_template=dict(
        type=PromptTemplate,
        template=dict(
            round=[
                dict(
                    role='HUMAN',
                    prompt='{problem}',  # Template for prompting the model
                ),
            ]
        ),
    ),
    retriever=dict(type=ZeroRetriever),
    inferencer=dict(type=GenInferencer),
)
```

3. Evaluation Configuration with LLM Judge

```python
eval_cfg = dict(
    evaluator=dict(
        type=GenericLLMEvaluator,  # Using LLM as evaluator
        prompt_template=dict(
            type=PromptTemplate,
            template=dict(
                begin=[
                    dict(
                        role='SYSTEM',
                        fallback_role='HUMAN',
                        prompt="You are a helpful assistant who evaluates the correctness and quality of models' outputs.",
                    )
                ],
                round=[
                    dict(role='HUMAN', prompt=YOUR_JUDGE_TEMPLATE),  # Template for the judge
                ],
            ),
        ),
        dataset_cfg=dict(
            type=CustomDataset,
            path='path/to/your/dataset',
            file_name='your_dataset.jsonl',
            reader_cfg=reader_cfg,
        ),
        judge_cfg=YOUR_JUDGE_MODEL_CONFIG,  # Configuration for the judge model
        dict_postprocessor=dict(type=generic_llmjudge_postprocess),  # Post-processing the judge's output
    ),
)
```

## Using CustomDataset with GenericLLMEvaluator

Here's how to set up a complete configuration for LLM judge evaluation:

```python
from mmengine.config import read_base
from opencompass.models import TurboMindModelwithChatTemplate
from opencompass.datasets import CustomDataset
from opencompass.evaluator import GenericLLMEvaluator
from opencompass.datasets import generic_llmjudge_postprocess
from opencompass.openicl.icl_prompt_template import PromptTemplate
from opencompass.openicl.icl_retriever import ZeroRetriever
from opencompass.openicl.icl_inferencer import GenInferencer

# Import your judge model configuration
with read_base():
    from opencompass.configs.models.qwen2_5.lmdeploy_qwen2_5_14b_instruct import (
        models as judge_model,
    )

# Define your judge template
JUDGE_TEMPLATE = """
Please evaluate whether the following response correctly answers the question.
Question: {problem}
Reference Answer: {answer}
Model Response: {prediction}

Is the model response correct? If correct, answer "A"; if incorrect, answer "B".
""".strip()

# Dataset reader configuration
reader_cfg = dict(input_columns=['problem'], output_column='answer')

# Inference configuration for the model being evaluated
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

# Evaluation configuration with LLM judge
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
                        prompt="You are a helpful assistant who evaluates the correctness and quality of models' outputs.",
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

# Dataset configuration
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

# Model configuration for the model being evaluated
models = [
    dict(
        type=TurboMindModelwithChatTemplate,
        abbr='model-to-evaluate',
        path='path/to/your/model',
        # ... other model configurations
    )
]

# Output directory
work_dir = './outputs/llm_judge_eval'
```

## GenericLLMEvaluator

The GenericLLMEvaluator is designed to use an LLM as a judge for evaluating model outputs. Key features include:

1. Flexible prompt templates for instructing the judge
2. Support for various judge models (local or API-based)
3. Customizable evaluation criteria through prompt engineering
4. Post-processing of judge outputs to extract structured evaluations

**Important Note**: The current generic version of the judge template only supports outputs in the format of "A" (correct) or "B" (incorrect), and does not support other output formats (like "CORRECT" or "INCORRECT"). This is because the post-processing function `generic_llmjudge_postprocess` is specifically designed to parse this format.

The evaluator works by:

1. Taking the original problem, reference answer, and model prediction
2. Formatting them into a prompt for the judge model
3. Parsing the judge's response to determine the evaluation result (looking for "A" or "B")
4. Aggregating results across the dataset

If you would like to see the full details of evaluation results, you can add `--dump-eval-details` to the command line when you start the job.
Example evaluation output:

```python
{
    'accuracy': 75.0,  # Percentage of responses judged as correct
    'details': [
        {
            'origin_prompt': """
            Please evaluate whether the following response correctly answers the question.
            Question: What is the capital of France?
            Reference Answer: Paris
            Model Response: Paris
            Is the model response correct? If correct, answer "A"; if incorrect, answer "B".
""",
            'gold': 'Paris',
            'prediction': 'A',
        },
        # ... more results
    ]
}
```

## Complete Example

For a complete working example, refer to the `eval_llm_judge.py` file in the examples directory, which demonstrates how to evaluate mathematical problem-solving using an LLM judge.
