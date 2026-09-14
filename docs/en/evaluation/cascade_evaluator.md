# Cascade Evaluation

## Introduction

Rule-based evaluation—regular-expression extraction, exact matching, or mathematical equivalence—is inexpensive, stable, and reproducible, but has poor recall when answers take varied forms. LLM judging is flexible, but consumes one model request per sample. `CascadeEvaluator` combines them: **first score every sample with a rule evaluator, then send samples marked incorrect by the rule to an LLM judge for review**. The final score combines both layers.

A typical use case is a mathematics dataset. When `MATHVerifyEvaluator` cannot extract `\boxed{}` from a prediction or symbolic equivalence fails, a judge reviews the answer against the question to avoid marking a correct answer wrong only because its form differs.

The `parallel` argument selects one of two modes:

- **Cascade mode** (`parallel=False`): send only rule-incorrect samples to the LLM. Final correct = rule correct ∪ (rule incorrect & LLM correct). The score can only increase relative to pure rule scoring, and judge request count is proportional to rule errors, making this the least expensive mode.
- **Parallel mode** (`parallel=True`, default): send every sample to the LLM. A sample is correct if either the rule or LLM says it is correct. This treats the rule as another opinion alongside the judge and is most permissive, but costs the same as pure LLM judging.

## Execution Mechanism

After the evaluation task calls `score()`, the evaluator performs:

1. **Per-sample rule evaluation:** apply the rule evaluator's `pred_postprocess` to each prediction, then score that sample and store the result in `rule_evaluation`. The log prints rule accuracy as `Rule-based evaluation: ...`.
2. **Collect samples for review:** cascade mode collects rule-incorrect samples; parallel mode collects all samples. `Samples requiring LLM evaluation (...)` in the log reports the count.
3. **Build the judging subset:** `select` samples from the original test set and append `prediction` and `reference` columns. The `llm_evaluator`'s `dataset_cfg` is cleared automatically and this subset is used directly, so the judge template can refer to every data column, including the question, reference, and model prediction.
4. **LLM judging:** call `llm_evaluator.score()`, which performs a complete `GenInferencer` run against the judge model. Results are written to `<result directory>_llm_judge_replica<N>.json`, where N is the repeated-run index.
5. **Decision and aggregation:** extract decisions from judge details, combine them according to the selected mode into final `accuracy`, and write `cascade_correct` for every sample.

Step 5 checks whether `prediction` / `llm_judge` in judge details equals `"A"` or starts with `"CORRECT"`, followed by a `correct` boolean and `score > 0.5`. Therefore, **the judge template must require only A / B** (or CORRECT / INCORRECT), consistent with [LLM as Judge](llm_judge.md). A paragraph of explanation from the judge is treated as incorrect.

## Configuration

`CascadeEvaluator` arguments:

| Argument | Type | Description |
| --- | --- | --- |
| `llm_evaluator` | required dict | LLM judge configuration, normally `GenericLLMEvaluator` |
| `rule_evaluator` | dict | Rule evaluator configuration, such as `MATHVerifyEvaluator` |
| `sample_score_fn` | Callable | Custom per-sample scoring function returning a dict with `correct`, or a boolean |
| `parallel` | bool, default `True` | `False` selects cascade mode; `True` selects parallel mode |

At least one of `rule_evaluator` and `sample_score_fn` must be supplied, otherwise initialization fails. Because the scoring workflow also calls the rule evaluator's `pred_postprocess`, practical configurations should always provide `rule_evaluator`.

The evaluation task supplies the test set required for LLM judging. `llm_evaluator.dataset_cfg` exists only to satisfy construction of `GenericLLMEvaluator`; cascade evaluation clears it and does not load the dataset again. `judge_cfg` specifies the judge model: an empty `dict()` reads `OC_JUDGE_MODEL` / `OC_JUDGE_API_KEY` / `OC_JUDGE_API_BASE`, or it can contain any local or API model configuration.

The following is a complete mathematics example using `MATHVerifyEvaluator` for the rule layer and environment variables for the judge:

```python
from mmengine.config import read_base

from opencompass.openicl.icl_prompt_template import PromptTemplate
from opencompass.openicl.icl_retriever import ZeroRetriever
from opencompass.openicl.icl_inferencer import GenInferencer
from opencompass.evaluator import (
    CascadeEvaluator,
    GenericLLMEvaluator,
    MATHVerifyEvaluator,
)
from opencompass.datasets import MATHDataset

with read_base():
    from opencompass.configs.models.qwen2_5.lmdeploy_qwen2_5_7b_instruct import (
        models as lmdeploy_qwen2_5_7b_instruct_model,
    )

reader_cfg = dict(input_columns=['problem'], output_column='solution')

infer_cfg = dict(
    prompt_template=dict(
        type=PromptTemplate,
        template=dict(
            round=[
                dict(
                    role='HUMAN',
                    prompt='{problem}\nReason step by step and put the final answer in \\boxed{}.',
                ),
            ]
        ),
    ),
    retriever=dict(type=ZeroRetriever),
    inferencer=dict(type=GenInferencer),
)

# The judge template permits only A / B
JUDGE_TEMPLATE = """Determine whether the predicted answer matches the reference answer.
Question: {problem}
Reference answer: {solution}
Predicted answer: {prediction}

Reply "A" if they match or "B" if they do not. Output nothing else.""".strip()

llm_judge_evaluator = dict(
    type=GenericLLMEvaluator,
    prompt_template=dict(
        type=PromptTemplate,
        template=dict(
            begin=[
                dict(
                    role='SYSTEM',
                    fallback_role='HUMAN',
                    prompt="You are an assistant responsible for judging the correctness of model output.",
                )
            ],
            round=[dict(role='HUMAN', prompt=JUDGE_TEMPLATE)],
        ),
    ),
    dataset_cfg=dict(
        type=MATHDataset,
        path='opencompass/math',
        file_name='test_prm800k_500.json',
    ),
    judge_cfg=dict(),  # Empty: read OC_JUDGE_* environment variables
)

eval_cfg = dict(
    evaluator=dict(
        type=CascadeEvaluator,
        llm_evaluator=llm_judge_evaluator,
        rule_evaluator=dict(type=MATHVerifyEvaluator),
        parallel=False,  # Cascade mode: send only rule-incorrect samples to judge
    ),
)

math_datasets = [
    dict(
        abbr='math_prm800k_500',
        type=MATHDataset,
        path='opencompass/math',
        file_name='test_prm800k_500.json',
        reader_cfg=reader_cfg,
        infer_cfg=infer_cfg,
        eval_cfg=eval_cfg,
    )
]

datasets = math_datasets
models = lmdeploy_qwen2_5_7b_instruct_model

work_dir = 'math_prm800k_500_cascade_evaluator'
```

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

Repository example [examples/eval_cascade_evaluator.py](https://github.com/open-compass/opencompass/blob/main/examples/eval_cascade_evaluator.py) demonstrates a complete cascade-evaluator configuration for MATH.
