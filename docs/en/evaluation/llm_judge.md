# LLM as Judge Evaluation

## Introduction

The GenericLLMEvaluator is particularly useful for scenarios where rule-based methods (like regular expressions) cannot perfectly judge outputs, such as:

- Cases where models output answer content without option identifiers
- Factual judgment datasets that are difficult to evaluate with rules
- Open-ended responses requiring complex understanding and reasoning
- Evaluation that requires a lot of rules to be designed

OpenCompass provides the GenericLLMEvaluator component to facilitate LLM-as-judge evaluations.

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

By default, OpenCompass uses these three environment variables. If a config file explicitly provides a concrete `judge_cfg`, these environment variables will not take effect.

### Using LLM for Evaluation via Configuration Files

Refer to the existing HLE config: [opencompass/configs/datasets/HLE/hle_llmverify_rawprompt_gen_0970dd.py](https://github.com/open-compass/opencompass/blob/main/opencompass/configs/datasets/HLE/hle_llmverify_rawprompt_gen_0970dd.py#L1-L80). This config uses `HLEDataset`, `RawPromptTemplate`, and `GenericLLMEvaluator`, and leaves `judge_cfg` empty so the judge model is read from `OC_JUDGE_MODEL` / `OC_JUDGE_API_KEY` / `OC_JUDGE_API_BASE`.

The components used by the config are imported as follows ([lines 1-6](https://github.com/open-compass/opencompass/blob/main/opencompass/configs/datasets/HLE/hle_llmverify_rawprompt_gen_0970dd.py#L1-L6)):

```python
from opencompass.openicl.icl_raw_prompt_template import RawPromptTemplate
from opencompass.openicl.icl_retriever import ZeroRetriever
from opencompass.openicl.icl_inferencer import GenInferencer
from opencompass.evaluator import GenericLLMEvaluator
from opencompass.datasets import generic_llmjudge_postprocess
from opencompass.datasets import HLEDataset
```

#### 1. Dataset Reader Configuration

`reader_cfg` defines which fields are passed to the evaluated model and which column is used as the reference answer. HLE uses `problem` as the question column and `answer` as the reference-answer column ([line 10](https://github.com/open-compass/opencompass/blob/main/opencompass/configs/datasets/HLE/hle_llmverify_rawprompt_gen_0970dd.py#L10)):

```python
math_reader_cfg = dict(input_columns=['problem'], output_column='answer')
```

#### 2. Inference Configuration

`infer_cfg` defines how dataset fields are converted into the evaluated model's input. This config uses `RawPromptTemplate` to place `{problem}` in the user message and asks the model to put its final answer in `\boxed{}` ([lines 12-21](https://github.com/open-compass/opencompass/blob/main/opencompass/configs/datasets/HLE/hle_llmverify_rawprompt_gen_0970dd.py#L12-L21)):

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

#### 3. Evaluation Configuration with LLM Judge

`GRADER_TEMPLATE` is the prompt template for the judge model. It references the original problem `{problem}`, the reference answer `{answer}`, and the model prediction `{prediction}` appended during evaluation ([lines 23-46](https://github.com/open-compass/opencompass/blob/main/opencompass/configs/datasets/HLE/hle_llmverify_rawprompt_gen_0970dd.py#L23-L46)):

The judge prompt used by `GenericLLMEvaluator` does not have a fixed whitelist of placeholders. When `RawPromptTemplate` is used, any `{column_name}` in `messages[*].content` is replaced by the field with the same name in the current judge-stage sample. If the field does not exist, the placeholder is left unchanged in the prompt.

The available fields usually come from two sources:

- Columns in the original evaluation dataset, such as HLE's `problem` and `answer`. If the template uses `{answer}`, the dataset loaded by `dataset_cfg` for the judge stage must contain an `answer` column.
- Columns appended by `GenericLLMEvaluator` before judging: `prediction` is the evaluated model's output; `reference` is the reference answer passed to the evaluator, usually from `reader_cfg.output_column` after `dataset_postprocessor` is applied; `obj_gold` uses the same reference-answer values as `reference`.

Therefore, the three placeholders in the HLE config come from:

- `{problem}` comes from the `problem` column in the original HLE dataset. This column is also declared in `math_reader_cfg.input_columns`, so it is used as an input field for the evaluated model.
- `{answer}` comes from the `answer` column in the original HLE dataset, which is the reference-answer column specified by `math_reader_cfg.output_column`.
- `{prediction}` is not a field in the original dataset. It is the evaluated model's output, appended by `GenericLLMEvaluator` before calling the judge model.

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

`math_eval_cfg` uses `GenericLLMEvaluator` to call the judge model and `generic_llmjudge_postprocess` to parse the judge output into an aggregatable result. `judge_cfg=dict()` means the judge model is not hard-coded in the config and is instead read from environment variables ([lines 48-68](https://github.com/open-compass/opencompass/blob/main/opencompass/configs/datasets/HLE/hle_llmverify_rawprompt_gen_0970dd.py#L48-L68)):

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

#### 4. Dataset Configuration

Finally, the dataset config combines `reader_cfg`, `infer_cfg`, and `eval_cfg` into `hle_datasets`, which OpenCompass can load ([lines 71-80](https://github.com/open-compass/opencompass/blob/main/opencompass/configs/datasets/HLE/hle_llmverify_rawprompt_gen_0970dd.py#L71-L80)):

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

## Using CustomDataset with GenericLLMEvaluator

### Dataset and Column Configuration

This section only covers data files and column configuration when using the built-in `CustomDataset`. For which fields a judge prompt can reference, see the `GenericLLMEvaluator` placeholder explanation above.

`CustomDataset` currently supports `.jsonl` and `.csv`. The data file should contain at least the original fields referenced by the judge template, for example:

- A problem or question
- A reference answer or gold standard

The reference-answer column name must match `reader_cfg.output_column`, for example `answer`, `gold`, or `target`. The model prediction does not need to be present in the original data file; it is generated during evaluation and then provided to the judge.

Example JSONL format:

```json
{"problem": "What is the capital of France?", "answer": "Paris"}
```

Example CSV format:

```text
problem,answer
"What is the capital of France?","Paris"
```

Here's how to set up a complete configuration for LLM judge evaluation:

```python
from mmengine.config import read_base
from opencompass.models import TurboMindModelwithChatTemplate
from opencompass.datasets import CustomDataset
from opencompass.evaluator import GenericLLMEvaluator
from opencompass.datasets import generic_llmjudge_postprocess
from opencompass.openicl.icl_raw_prompt_template import RawPromptTemplate
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
        type=RawPromptTemplate,
        messages=[
            dict(role='user', content='{problem}'),
        ],
    ),
    retriever=dict(type=ZeroRetriever),
    inferencer=dict(type=GenInferencer),
)

# Evaluation configuration with LLM judge
eval_cfg = dict(
    evaluator=dict(
        type=GenericLLMEvaluator,
        prompt_template=dict(
            type=RawPromptTemplate,
            messages=[
                dict(role='system', content="You are a helpful assistant who evaluates the correctness and quality of models' outputs."),
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

## Example Scripts

For a runnable example of the generic LLM judge evaluator, see `examples/eval_llm_judge.py`. This script uses `CustomDataset` and `GenericLLMEvaluator` to demonstrate inference, judge-model evaluation, and result aggregation on mathematical problems.

## CascadeEvaluator

Rule-based and LLM evaluation can be cascaded: rules run first, then rule-incorrect samples are sent to an LLM judge for review. See [Cascade Evaluation](cascade_evaluator.md) for `CascadeEvaluator` modes, decision rules, caching, and rerun behavior.
