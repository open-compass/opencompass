# Adding Evaluators, Postprocessors, and Summarizers

First identify the layer to extend: a postprocessor turns text into a normalized answer, an Evaluator computes metrics from predictions and references, and a Summarizer organizes existing metrics from multiple Datasets.

## Postprocessors

A postprocessor should be deterministic and free of side effects, with explicit handling for empty replies, multiple answers, malformed formats, and outliers. It is normally configured in Dataset `eval_cfg`:

```python
eval_cfg = dict(
    pred_postprocessor=dict(type=my_pred_postprocess),
    dataset_postprocessor=dict(type=my_reference_postprocess),
    evaluator=dict(type=MyEvaluator),
)
```

## Evaluator

### Implementation Requirements

An Evaluator subclasses `BaseEvaluator` (`opencompass.openicl.icl_evaluator`) and is registered with `ICL_EVALUATORS`. **The only required method is `score()`.** If the constructor accepts `pred_postprocessor` and forwards it to `super().__init__`, the framework applies it automatically before each scoring run. A minimal skeleton is:

```python
from opencompass.openicl.icl_evaluator import BaseEvaluator
from opencompass.registry import ICL_EVALUATORS


@ICL_EVALUATORS.register_module()
class MyEvaluator(BaseEvaluator):

    def __init__(self, pred_postprocessor=None):
        super().__init__(pred_postprocessor=pred_postprocessor)

    def score(self, predictions, references, test_set=None):
        details = []
        correct = 0
        for pred, ref in zip(predictions, references):
            is_correct = str(pred).strip() == str(ref).strip()
            correct += int(is_correct)
            details.append(
                dict(pred=pred, answer=ref, correct=is_correct))
        return {'accuracy': 100 * correct / len(predictions),
                'details': details}
```

The evaluation task does not call `score()` directly. It goes through base-class `evaluate(k, n, original_dataset, **score_kwargs)`, which:

1. Assembles arguments from the `score()` signature: `predictions`, `references`, `test_set`, and any test-set columns sharing names with signature parameters. To use question metadata, declare its column name directly in the signature.
2. Splits predictions into n runs according to `n`, applies `pred_postprocessor` to each, and scores them independently.
3. Averages numeric metrics across runs and appends `(n runs average)` to metric names when `n > 1`.
4. Removes and aggregates `details`, grouping them by sample across runs.

### `score()` Return Format

- Return `dict[metric name, value]`, where values must be `int` / `float`. Summarization retains numeric results only and silently drops other types. A result containing an `'error'` key is skipped entirely and logged.
- Metric names should be stable and readable. Common names such as `accuracy`, `exact_match`, `f1`, and `rouge1` sort before uncommon names in summary tables.
- Optionally return `'details'` as `list[dict]`. The base class aggregates it across repeated runs and writes it to the result file for sample-by-sample review with `--dump-eval-details`. If each detail contains a `correct`, `is_correct`, or `cascade_correct` boolean, the base class also computes cross-run metrics such as G-Pass@k and mG-Pass@k when `n > 1` and `k > 1`.
- A custom class should not override `evaluate()` unless repeated-run partition semantics genuinely need to change.

Return stable metric names and test normal input, empty input, parse failure, boundary values, and multiple references. An evaluator accessing the network, invoking a Judge, or executing code must define timeouts, error recording, and isolation.

## Summarizer

### Prefer `summary_groups` Configuration

When a summarizer configuration omits `type`, it uses `DefaultSummarizer`. It collects model-dataset metric files from `<work_dir>/results/`, computes group aggregates, and writes txt / csv / md tables under `<work_dir>/summary/`. `dataset_abbrs` controls row order and can include group names and empty strings `''` as visual separators.

Most requirements—grouping, averaging, and weighting—should be expressed through `summary_groups`, not a new Summarizer class.

### Summary Group Fields

| Field | Type | Description |
| --- | --- | --- |
| `name` | str | Group name, shown as a row in the summary table |
| `subsets` | list | Dataset-abbreviation strings or `[abbr, metric]` pairs; **the two forms cannot be mixed** |
| `metric` | str | Explicit metric name to aggregate |
| `weights` | dict | `{subset abbr: weight}` enabling weighted average |
| `std` / `sum` / `harmonic_mean` | bool | Use standard deviation, sum, or harmonic mean, respectively |
| `transforms` | dict | `{subset abbr: 'expression'}` transforming a score before aggregation; `x` is the original value |

Aggregation precedence is: explicit `metric` > the method corresponding to `std` / `sum` / `weights` / `harmonic_mean` > simple average (macro average).

**If any subset result is missing, the whole group is marked `error: missing metrics`; no partial aggregate is produced.** This guarantees that group scores always use all subsets and helps identify missing datasets during debugging.

### When `subsets` Can Contain Dataset Names Directly

When every `subsets` item is a string, the group averages each metric common to all subsets and adds one aggregate row. That row uses each subset's first metric, after the main-metric priority ordering `accuracy` > `exact_match` > ... Therefore:

- If all subsets use the **same main metric**, such as `accuracy` for all 57 MMLU subjects, dataset names alone produce the intended average.
- If subset metric conventions differ, or a subset has multiple metrics and the default main metric is not the desired one, use `[subset abbr, metric name]` pairs explicitly. The group then aggregates only the selected metrics and does not average common metrics.

`transforms` can normalize scales before aggregation, such as multiplying a 0–1 score by 100. It executes expressions through `eval` and must appear only in trusted configuration.

### Weighted Average (`weights`)

`weights` assigns each subset a weight. The aggregate row is named `weighted_average` and computes `Σ(w·x) / Σw`; zero-weight subsets are skipped so their NaN values do not contaminate the result. A typical repository pattern reports the same subsets under two conventions:

```python
# Excerpt from opencompass/configs/summarizers/groups/mmlu.py
mmlu_summary_groups.append({'name': 'mmlu', 'subsets': _mmlu_all})
mmlu_summary_groups.append(
    {'name': 'mmlu-weighted', 'subsets': _mmlu_all, 'weights': _mmlu_weights})
```

`mmlu` is a macro average across subjects, while `mmlu-weighted` weights by official subject sample count, equivalent to sample-size weighting. RewardBench-style configurations instead use official proportions so their total matches the official leaderboard. A `weights` key is a subset abbreviation and also supports the `subset@metric` form.

### When to Write a Custom Summarizer Class

Add a Summarizer only when the default table cannot express the grouping or aggregation convention. Subclass `DefaultSummarizer` and override `summarize(output_path, time_str)`; repository examples include `CircularSummarizer` and `MultiFacetedSummarizer`. Set `type=MySummarizer` in configuration. An aggregate score must document:

- Whether it is a macro average or sample-count weighted.
- How missing subsets are handled.
- Whether metric direction and scaling agree.
- How repeated runs are aggregated.

Prefer expressing groups through configuration under `opencompass/configs/summarizers/`. Implement and register a class only for genuinely new behavior. Validate it against a small fixed result set rather than depending on full model inference.
