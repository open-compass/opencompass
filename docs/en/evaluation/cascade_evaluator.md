# Cascade Evaluation

## Introduction

Rule-based evaluation—regular-expression extraction, exact matching, or mathematical equivalence—is inexpensive, stable, and reproducible, but has poor recall when answers take varied forms. LLM judging is flexible, but consumes one model request per sample. `CascadeEvaluator` combines them: **first score every sample with a rule evaluator, then send samples marked incorrect by the rule to an LLM judge for review**. The final score combines both layers.

A typical use case is a mathematics dataset. When `MATHVerifyEvaluator` cannot extract `\boxed{}` from a prediction or symbolic equivalence fails, a judge reviews the answer against the question to avoid marking a correct answer wrong only because its form differs.

The `parallel` argument selects one of two modes:

- **Cascade mode** (`parallel=False`): send only rule-incorrect samples to the LLM. Final correct = rule correct ∪ (rule incorrect & LLM correct). The score can only increase relative to pure rule scoring, and judge request count is proportional to rule errors, making this the least expensive mode.
- **Parallel mode** (`parallel=True`, default): send every sample to the LLM. A sample is correct if either the rule or LLM says it is correct. This treats the rule as another opinion alongside the judge and is most permissive, but costs the same as pure LLM judging.

## Execution Mechanism

After the evaluation task calls `score()`, the evaluator performs:

1. **Per-sample initial evaluation:** if `rule_evaluator` is provided, apply its `pred_postprocess` before scoring the sample with the rule evaluator or `sample_score_fn`; if only `sample_score_fn` is provided, score the raw prediction directly. The result is stored in `rule_evaluation`. The log prints the initial evaluation accuracy as `Rule-based evaluation: ...`.
2. **Collect samples for review:** cascade mode collects samples marked incorrect by the initial evaluation; parallel mode collects all samples. `Samples requiring LLM evaluation (...)` in the log reports the count.
3. **Build the judging subset:** `select` samples from the original test set and append `prediction` and `reference` columns. The `llm_evaluator`'s `dataset_cfg` is cleared automatically and this subset is used directly, so the judge template can refer to every data column, including the question, reference, and model prediction.
4. **LLM judging:** call `llm_evaluator.score()`, which performs a complete `GenInferencer` run against the judge model. Results are written to `<result directory>_llm_judge_replica<N>.json`, where N is the repeated-run index.
5. **Decision and aggregation:** extract decisions from judge details, combine them according to the selected mode into final `accuracy`, and write `cascade_correct` for every sample.

Step 5 checks whether `prediction` / `llm_judge` in judge details equals `"A"` or starts with `"CORRECT"`, followed by a `correct` boolean and `score > 0.5`. Therefore, **the judge template must require only A / B** (or CORRECT / INCORRECT), consistent with [LLM as Judge](llm_judge.md). A paragraph of explanation from the judge is treated as incorrect.

## Configuration

`CascadeEvaluator` arguments:

| Argument          | Type                                      | Description                                                                      |
| ----------------- | ----------------------------------------- | -------------------------------------------------------------------------------- |
| `llm_evaluator`   | required dict                             | LLM judge configuration, normally `GenericLLMEvaluator`                          |
| `rule_evaluator`  | dict                                      | Rule evaluator configuration, such as `MATHVerifyEvaluator`                      |
| `sample_score_fn` | `Callable[[str, str, Any], dict \| bool]` | Custom per-sample scoring function returning a dict with `correct`, or a boolean |
| `parallel`        | bool, default `True`                      | `False` selects cascade mode; `True` selects parallel mode                       |

At least one of `rule_evaluator` and `sample_score_fn` must be supplied, otherwise initialization fails. If `rule_evaluator` is provided, its `pred_postprocess` is applied before scoring; if only `sample_score_fn` is provided, the raw prediction is passed to `sample_score_fn` directly. If both are provided, `sample_score_fn` takes precedence for per-sample scoring and receives the prediction after `rule_evaluator.pred_postprocess`.

`sample_score_fn` is called as `sample_score_fn(prediction, reference, test_item)`: `prediction` is one model prediction, `reference` is the corresponding reference answer, and `test_item` is the original test sample (`None` when no `test_set` is passed). If it returns a dict, the dict should contain at least `correct`, for example `{'correct': True, 'pred': prediction, 'answer': reference}`; extra fields are kept in per-sample evaluation details. If it returns a non-dict value, the framework converts `bool(result)` to `correct` and automatically adds `pred` and `answer`.

The evaluation task supplies the test set required for LLM judging. `llm_evaluator.dataset_cfg` exists only to satisfy construction of `GenericLLMEvaluator`; cascade evaluation clears it and does not load the dataset again. `judge_cfg` specifies the judge model: an empty `dict()` reads `OC_JUDGE_MODEL` / `OC_JUDGE_API_KEY` / `OC_JUDGE_API_BASE`, or it can contain any local or API model configuration.

The repository's [AIME 2026 cascade evaluation config](https://github.com/open-compass/opencompass/blob/main/opencompass/configs/datasets/aime2026/aime2026_cascade_eval_rawprompt_gen_0970dd.py) is a complete example: the rule layer uses `MATHVerifyEvaluator`, rule-incorrect samples are reviewed by `GenericLLMEvaluator`, and the judge model is configured through environment variables.

```python
from opencompass.openicl.icl_raw_prompt_template import RawPromptTemplate
from opencompass.openicl.icl_retriever import ZeroRetriever
from opencompass.openicl.icl_inferencer import GenInferencer
from opencompass.datasets import CustomDataset
from opencompass.datasets import generic_llmjudge_postprocess
from opencompass.evaluator import (
    CascadeEvaluator,
    GenericLLMEvaluator,
    MATHVerifyEvaluator
)

aime2026_reader_cfg = dict(input_columns=['problem'], output_column='answer')

aime2026_infer_cfg = dict(
    prompt_template=dict(
        type=RawPromptTemplate,
        messages=[
            {'role': 'user', 'content': '{problem}\nRemember to put your final answer within \\boxed{}.'},
        ],
    ),
    retriever=dict(type=ZeroRetriever),
    inferencer=dict(type=GenInferencer),
)

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

cascade_evaluator = dict(
    type=CascadeEvaluator,
    rule_evaluator=dict(
        type=MATHVerifyEvaluator,
    ),
    llm_evaluator=dict(
        type=GenericLLMEvaluator,
        prompt_template=dict(
            type=RawPromptTemplate,
            messages=[
                {'role': 'system', 'content': "You are a helpful assistant who evaluates the correctness and quality of models' outputs."},
                {'role': 'user', 'content': GRADER_TEMPLATE},
            ],
        ),
        dataset_cfg=dict(
            type=CustomDataset,
            path='opencompass/aime2026',
            reader_cfg=aime2026_reader_cfg,
        ),
        judge_cfg=dict(),
        dict_postprocessor=dict(type=generic_llmjudge_postprocess),
    ),
    parallel=False,
)
aime2026_eval_cfg = dict(
    evaluator=cascade_evaluator,
)

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

Here, `aime2026_reader_cfg` selects the `problem` column and the `answer` reference column. `GRADER_TEMPLATE` inserts the question, reference, and prediction through `{problem}`, `{answer}`, and `{prediction}`, and asks the judge to return `A` / `B` (the decision logic also accepts `CORRECT` / `INCORRECT`). In `cascade_evaluator`, `rule_evaluator` is the first-layer mathematical-equivalence check and `llm_evaluator` is the second-layer review. `parallel=False` means only samples marked incorrect by the first layer are reviewed. `dict_postprocessor` uses `generic_llmjudge_postprocess` to turn judge output into evaluation details. An empty `judge_cfg` reads the judge model configuration from the `OC_JUDGE_*` environment variables.

## Evaluation Output

In addition to `accuracy`, the result contains `cascade_stats` and per-sample `details`:

```python
{
    'accuracy': 85.0,
    'cascade_stats': {
        'total_samples': 100,
        'rule_correct': 70,
        'rule_accuracy': 70.0,
        'llm_evaluated': 30,
        'llm_correct': 15,
        'llm_accuracy': 50.0,
        'final_correct': 85,
        'final_accuracy': 85.0,
        'parallel_mode': False,
    },
    'details': [
        # ... one entry per sample
    ],
}
```

The fields mean:

- `rule_accuracy`: pure-rule accuracy and the lower bound of the final score in cascade mode.
- `llm_evaluated`: samples actually sent to the judge (rule-error count in cascade mode; all samples in parallel mode).
- `llm_accuracy`: judge accuracy on those samples, useful for estimating rule-evaluator false negatives.
- `final_accuracy`: final reported metric after combining the two layers.

Each sample's `details` structure is:

```python
{
    'rule_evaluation': {'correct': False, ...},                # Rule-layer detail
    'llm_evaluation': {'prediction': 'A', 'llm_correct': True, ...},  # Judge detail, if any
    'cascade_correct': True,                                   # Final combined decision
}
```

Add `--dump-eval-details` when launching the task to write these details and review disagreements between the rule and judge sample by sample.

## Result Cache and Reruns

Judge results are stored in `<result directory>_llm_judge_replica<N>.json`. If the file exists on a later evaluation, it is loaded and the judge is not requested again. If its sample count differs from the current review set, evaluation raises an error and asks you to delete the cache. Changing the evaluated model, rule evaluator, or answer postprocessing can all change the rule-error set and invalidate the cache.

Repeated runs (dataset replicas) use independently numbered cache files.

## When to Use Cascade Evaluation

- A rule evaluator exists but has insufficient recall, such as mathematical equivalence or open-ended QA: use **cascade mode**, spending judge cost only on rule failures.
- Rules and judge should cover for each other and correctness is their union: use **parallel mode**.
- Rules are fully reliable, as in ordinary multiple choice: use the rule evaluator directly, without a judge.
- No judge service is available or cost must be strictly controlled: use pure rule evaluation.

Compared with pure LLM judging, cascade scores can be decomposed into a rule score and judge correction, as shown in `cascade_stats`. External reports should include both layers so they remain comparable with results produced by pure rules or a pure judge.

## Complete Example

See [opencompass/configs/datasets/aime2026/aime2026_cascade_eval_rawprompt_gen_0970dd.py](https://github.com/open-compass/opencompass/blob/main/opencompass/configs/datasets/aime2026/aime2026_cascade_eval_rawprompt_gen_0970dd.py) for the complete config.
