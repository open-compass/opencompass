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

### Using LLM for Evaluation via Command Line

Some datasets in OpenCompass already include LLM judge configurations.
You need to use a model service (such as OpenAI or DeepSeek's official API) or start a model service locally using tools like LMDeploy, vLLM, or SGLang.

Then, you can set the environment variables for the evaluation service and evaluate models using the following commands:

```bash
export OC_JUDGE_MODEL=Qwen/Qwen2.5-32B-Instruct
export OC_JUDGE_API_KEY=sk-1234
export OC_JUDGE_API_BASE=http://172.30.56.1:4000/v1
```

Note that by default, OpenCompass will use these three environment variables, but if you use configuration files to configure the evaluation service, these environment variables will not take effect.

### Using LLM for Evaluation via Configuration Files

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

## CascadeEvaluator

OpenCompass also provides a CascadeEvaluator that combines the strengths of rule-based evaluation and LLM-based evaluation. The cascade evaluator has two modes:

1. **Cascade Mode (parallel=False)**: First evaluates all samples with a rule-based evaluator, then only sends samples that were deemed incorrect by the rule-based evaluation to an LLM judge for re-evaluation. This approach reduces reliance on LLM judgments while maintaining accuracy, thus lowering evaluation costs and time.

2. **Parallel Mode (parallel=True)**: Evaluates all samples with both the rule-based evaluator and LLM judge, then considers a sample correct if either method marks it as correct. This approach can increase the leniency of evaluation but may result in higher costs since all samples require LLM evaluation.

### Configuring CascadeEvaluator

Here's an example of how to configure the CascadeEvaluator:

```python
# Define a rule-based evaluator
rule_evaluator = dict(type=MATHEvaluator)

# Define an LLM judge evaluator
llm_judge_evaluator = dict(
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
                dict(role='HUMAN', prompt=YOUR_JUDGE_TEMPLATE),
            ],
        ),
    ),
    dataset_cfg=dict(
        type=YourDataset,
        path='path/to/your/dataset',
        reader_cfg=reader_cfg,
    ),
    judge_cfg=dict(),  # Can use environment variables to configure the judge model
)

# Configure cascade evaluator (cascade mode)
cascade_evaluator = dict(
    type=CascadeEvaluator,
    llm_evaluator=llm_judge_evaluator,
    rule_evaluator=rule_evaluator,
    parallel=False  # Cascade mode
)

# For parallel mode, set parallel=True
parallel_evaluator = dict(
    type=CascadeEvaluator,
    llm_evaluator=llm_judge_evaluator,
    rule_evaluator=rule_evaluator,
    parallel=True  # Parallel mode
)

# Use the cascade evaluator in your dataset evaluation config
eval_cfg = dict(evaluator=cascade_evaluator)
```

### Evaluation Results

The cascade evaluator outputs detailed evaluation statistics including:

- Accuracy of the rule-based evaluation
- Accuracy of the LLM evaluation (for samples that failed rule-based evaluation in cascade mode)
- Final combined accuracy

Example output:

```python
{
    'accuracy': 85.0,  # Final accuracy
    'cascade_stats': {
        'total_samples': 100,
        'rule_correct': 70,  # Number of samples correct by rule evaluation
        'rule_accuracy': 70.0,  # Accuracy of rule evaluation
        'llm_evaluated': 30,  # Number of samples evaluated by LLM (failed samples in cascade mode)
        'llm_correct': 15,  # Number of samples correct by LLM evaluation
        'llm_accuracy': 50.0,  # Accuracy of LLM evaluation
        'final_correct': 85,  # Total correct samples
        'final_accuracy': 85.0,  # Final accuracy
        'parallel_mode': False,  # Whether parallel mode was used
    },
    'details': [
        # Detailed evaluation results for each sample
    ]
}
```

The cascade evaluator is particularly useful for:

1. Scenarios that require balancing evaluation cost and accuracy
2. Cases where rule-based evaluators are available but might not be comprehensive
3. Evaluation tasks that need more nuanced judgment for edge cases

## Complete Example

For a complete working example using GenericLLMEvaluator
, refer to the `eval_llm_judge.py` file in the examples directory, which demonstrates how to evaluate mathematical problem-solving .

For a complete working example using CascadeEvaluator, refer to the `eval_cascade_evaluator.py` file in the examples directory, which demonstrates how to evaluate mathematical problem-solving .
