# Adding Postprocessors, Evaluators, and Summarizers

First identify the layer to extend: a postprocessor turns text into a normalized answer, an Evaluator computes metrics from predictions and references, and a Summarizer organizes existing metrics from multiple Datasets.

## Postprocessors

A postprocessor should be deterministic and free of side effects, with explicit handling for empty replies, multiple answers, malformed formats, and outliers. For a typical Dataset, place postprocessors at the top level of `eval_cfg`: `pred_postprocessor` processes model predictions before they enter the Evaluator, while `dataset_postprocessor` processes reference answers in the test set.

```python
eval_cfg = dict(
    pred_postprocessor=dict(type=my_pred_postprocess),
    dataset_postprocessor=dict(type=my_reference_postprocess),
    evaluator=dict(type=MyEvaluator),
)
```

You can also configure `pred_postprocessor` inside `eval_cfg.evaluator`. In this case, the Evaluator constructor must accept the argument and pass it to `BaseEvaluator`. The base class applies it before calling `score()` for each repeated run:

```python
eval_cfg = dict(
    evaluator=dict(
        type=MyEvaluator,
        pred_postprocessor=dict(type='my_pred_postprocess'),
    ),
)
```

These two configurations run at different stages and are applied sequentially; one does not override the other. Do not configure the same postprocessor in both places, or predictions will be processed twice. Unless the postprocessing behavior must be tied to a particular Evaluator, prefer the first, Dataset-level configuration.

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

### Sources of `score()` Arguments

The arguments accepted by `score()` are not limited to `predictions` and `references`, but every argument must be a field that the evaluation task can provide. `OpenICLEvalTask` first collects fields from the prediction file, then adds or overwrites `predictions`, `references`, `test_set`, and `origin_prompt`. Finally, it selects fields whose names match the `score()` signature and passes them to the method.

Do not use `**kwargs` in `score()`. The current implementation identifies it as a parameter named `kwargs`, but the evaluation task does not provide such a field, so argument assembly fails. Likewise, do not declare a column name that exists only in the Dataset. The evaluation task does not automatically expand arbitrary Dataset columns. To access questions, options, test cases, or other metadata, declare `test_set` and read the fields from it.

Common arguments are listed below. The evaluation task adds or overwrites the first four; the remaining arguments are available only when the inferencer writes fields with those names to the prediction file.

| Argument           | Meaning                                                                                                                                                                                |
| ------------------ | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `predictions`      | A list of model predictions. For generation tasks, these values have passed through any configured model-level, Dataset-level, and Evaluator-level postprocessors. If each sample returns multiple candidates, the value may be a list of lists. |
| `references`       | A list of reference answers from the test-set column specified by `reader_cfg.output_column`; `None` when `output_column` is not configured.                                           |
| `test_set`         | The current test set as a `datasets.Dataset`, after the optional `dataset_postprocessor` has run. Dataset fields not passed separately should be read from this object.                |
| `origin_prompt`    | The original prompt or message written to the prediction file during inference. If absent, the evaluation task supplies a list of `None` values aligned with the predictions.          |
| `gold`             | The gold-answer field written by some inferencers. Unlike `references`, it is not always available; declare it only when the prediction file contains the field.                       |
| `steps`            | Intermediate steps written by the prediction file or a custom inference flow, commonly used to evaluate both final answers and reasoning steps.                                        |
| `res_length`       | Generated-response length statistics, usually written by a generation inferencer when response-length dumping is enabled.                                                              |
| `all_input_length` | Total input prompt or message length, usually used with `res_length` to analyze input and output lengths.                                                                              |
| `ppl`              | PPL/perplexity-related inference results, usually produced by a PPL inferencer or a custom prediction file.                                                                            |
| `token_len`        | Token counts paired with `ppl`, used to normalize PPL-based metrics by token count.                                                                                                    |
| `loss`             | Loss values, commonly used by loss-based metrics such as BPC.                                                                                                                          |
| `total_chr_num`    | Character counts paired with `loss`, commonly used to calculate bits per character.                                                                                                    |
| `mink`             | Min-K probability statistics used by the corresponding Min-K evaluator.                                                                                                                |
| `prompt`           | A prompt field in the prediction file, recorded by some PPL or conditional-probability inference flows.                                                                                |
| `choices`          | Candidate choices written by a conditional-probability inference flow.                                                                                                                 |
| `pred_label`       | The predicted label selected from scores by a conditional-probability inference flow.                                                                                                  |

If a custom inferencer writes other keys to the prediction file, `score()` may declare arguments with matching names. Because these fields depend on the inference flow, verify that the prediction file contains them before use.

After assembling the arguments, the evaluation task calls base-class `evaluate(k, n, original_dataset, **score_kwargs)` instead of invoking `score()` directly. The base class then:

1. Divides repeated inference results for the same Dataset into `n` independent batches, each containing the arguments for one complete run over the Dataset.
2. Applies the Evaluator-level `pred_postprocessor` to each batch of predictions, then calls `score()`.
3. Collects numeric metrics from all batches. When `n > 1`, it calculates their means and appends `(n runs average)` to their names. However, the current implementation returns these aggregated results only when every `score()` call returns complete, non-empty `details`; otherwise, it returns the last batch's result.
4. Removes and aggregates `details`, grouping entries for the same sample across runs.

### `score()` Return Format

- Return `dict[metric name, value]`, where values must be `int` / `float`. Summarization retains numeric results only and silently drops other types. A result containing an `'error'` key is skipped entirely and logged.
- Metric names should be stable and readable. Common names such as `accuracy`, `exact_match`, `f1`, and `rouge1` sort before uncommon names in summary tables.
- `'details'` (`list[dict]`) is optional at the interface level, but the current implementation has a limitation: when `n > 1` and the cross-run mean metrics are required, every `score()` call must return complete, non-empty `details` aligned one-to-one with the samples in the current batch. Otherwise, only the last batch's result is returned. The base class aggregates `details` across repeated runs and writes them to the result file for sample-by-sample review with `--dump-eval-details`. Individual entries need not include a correctness field; `correct`, `is_correct`, or `cascade_correct` booleans are required only to calculate cross-run metrics such as G-Pass@k and mG-Pass@k.
- A custom class should not override `evaluate()` unless repeated-run partition semantics genuinely need to change.

Return stable metric names and test normal input, empty input, parse failure, boundary values, and multiple references. An evaluator accessing the network, invoking a Judge, or executing code must define timeouts, error recording, and isolation.

## Summarizer

### Complete Example: InverseIFEval

When a summarizer configuration omits `type`, it uses `DefaultSummarizer`. It reads numeric metrics for each Dataset from `<work_dir>/results/`, generates group metrics according to `summary_groups`, and writes tables under `<work_dir>/summary/`.

The following self-contained example is based on the repository's InverseIFEval configuration and extracts its overall score, Chinese group, and macro average. The actual group definitions are in [`opencompass/configs/summarizers/groups/inverse_ifeval.py`](https://github.com/open-compass/opencompass/blob/main/opencompass/configs/summarizers/groups/inverse_ifeval.py):

```python
inverse_ifeval_instruction_type_abbrs = [
    'QC', 'ITF', 'CC', 'CCF', 'DIA', 'II', 'MIM', 'CA'
]
inverse_ifeval_language_abbrs = ['zh', 'en']

inverse_ifeval_subsets = [
    f'InverseIFEval_{language}_{instruction_type}'
    for language in inverse_ifeval_language_abbrs
    for instruction_type in inverse_ifeval_instruction_type_abbrs
]

# Number of samples per instruction type; half are Chinese and half English.
inverse_ifeval_type_counts = {
    'QC': 90,
    'ITF': 86,
    'CC': 198,
    'CCF': 82,
    'DIA': 186,
    'II': 154,
    'MIM': 108,
    'CA': 108,
}
inverse_ifeval_weights = {
    f'InverseIFEval_{language}_{instruction_type}':
    inverse_ifeval_type_counts[instruction_type] // 2
    for language in inverse_ifeval_language_abbrs
    for instruction_type in inverse_ifeval_instruction_type_abbrs
}

inverse_ifeval_summary_groups = [
    # Overall average weighted by the number of samples in each subset.
    dict(
        name='InverseIFEval',
        subsets=[[subset, 'accuracy'] for subset in inverse_ifeval_subsets],
        weights=inverse_ifeval_weights,
    ),
    # Chinese subsets only, weighted by sample count.
    dict(
        name='InverseIFEval_zh',
        subsets=[[
            f'InverseIFEval_zh_{instruction_type}', 'accuracy'
        ] for instruction_type in inverse_ifeval_instruction_type_abbrs],
        weights={
            f'InverseIFEval_zh_{instruction_type}':
            inverse_ifeval_type_counts[instruction_type] // 2
            for instruction_type in inverse_ifeval_instruction_type_abbrs
        },
    ),
    # Macro average in which all subsets have equal weight.
    dict(
        name='InverseIFEval_macro',
        subsets=[[subset, 'accuracy'] for subset in inverse_ifeval_subsets],
    ),
]
```

The corresponding main summarizer configuration imports these groups and controls the contents and order of the output table. The complete repository version is in `opencompass/configs/summarizers/inverse_ifeval.py`:

```python
from mmengine.config import read_base

with read_base():
    from .groups.inverse_ifeval import (inverse_ifeval_subsets,
                                        inverse_ifeval_summary_groups)

summarizer = dict(
    dataset_abbrs=[
        ['InverseIFEval', 'weighted_average'],
        ['InverseIFEval_zh', 'weighted_average'],
        ['InverseIFEval_macro', 'naive_average'],
        *[[subset, 'accuracy'] for subset in inverse_ifeval_subsets],
    ],
    summary_groups=inverse_ifeval_summary_groups,
)
```

This configuration is processed as follows:

1. Each `[Dataset abbr, metric]` pair in `subsets` selects one input value exactly. For example, `['InverseIFEval_zh_QC', 'accuracy']` reads `accuracy` from the result file for `InverseIFEval_zh_QC`. These pairs **select aggregation inputs**.
2. `name` becomes the abbreviation of a newly generated group. A single group name can have one or more aggregate metrics.
3. When `weights` is present, the result is `Σ(w·x) / Σw`, and the default metric name is `weighted_average`. Here, the weights are subset sample counts, so `InverseIFEval` is an average over all samples.
4. Without `weights` or another aggregation method, the result is the arithmetic mean of all input values, and the default metric name is `naive_average`. Thus, `InverseIFEval_macro` is a macro average that gives equal weight to all 16 subsets.
5. After `summary_groups` has been calculated, `dataset_abbrs` selects rows from the original Dataset metrics and newly generated group metrics for the output table.

#### How `dataset_abbrs` Controls the Output Table

When `dataset_abbrs` is configured, each list item corresponds to one output-table row, and list order determines row order:

- `[abbr, metric]` selects one row exactly. For example, `['InverseIFEval', 'weighted_average']` displays the `weighted_average` for the `InverseIFEval` group, while `['InverseIFEval_zh_QC', 'accuracy']` displays the original subset's `accuracy`. These pairs **select aggregation outputs**; they neither trigger nor change aggregation.
- A bare `abbr` string displays the first metric after sorting that abbreviation's metrics. Metrics are ordered according to [`METRIC_WHITELIST`](https://github.com/open-compass/opencompass/blob/main/opencompass/summarizers/default.py#L19), with metrics outside the list placed afterward. This shorthand therefore depends on default ordering and is less explicit than a pair.
- An empty string `''` does not match a result, so the table receives a placeholder row with an empty name and `-` in the other cells. It can serve as a visual separator.
- If the requested abbr or metric does not exist, the table likewise contains a row of `-` values instead of falling back to another metric. If the abbr and metric exist but a particular model has no corresponding result, only that model's cell contains `-`.

For `['InverseIFEval', 'weighted_average']`, lookup proceeds as follows:

1. `DefaultSummarizer` reads the original Dataset results and then processes each entry in `inverse_ifeval_summary_groups`.

2. It finds the group with `name='InverseIFEval'` and uses `name` to create a new group abbreviation, `InverseIFEval`.

3. Because the group has `weights` and no explicit `metric`, the weighted aggregate uses the default metric name `weighted_average`. After aggregation, the internal result is equivalent to:

   ```python
   parsed_results[model_abbr]['InverseIFEval']['weighted_average'] = score
   ```

4. While generating the table, `['InverseIFEval', 'weighted_average']` first looks up the group abbreviation using its first element, then looks up the metric within that group using its second element, and writes the resulting score as one row.

The first element can therefore refer either to an original Dataset abbreviation from the main configuration or to a group abbreviation created by `summary_groups[*].name`. The second element must be a metric that the selected abbreviation actually contains.

After expansion, the example's `dataset_abbrs` displays three aggregate rows followed by the `accuracy` of the 16 original subsets:

```text
dataset                    metric
InverseIFEval              weighted_average
InverseIFEval_zh           weighted_average
InverseIFEval_macro        naive_average
InverseIFEval_zh_QC        accuracy
InverseIFEval_zh_ITF       accuracy
...                        ...
InverseIFEval_en_CA        accuracy
```

If `dataset_abbrs` is omitted entirely, `DefaultSummarizer` first outputs every numeric metric for each Dataset in the order of `datasets` in the main configuration, then appends every group metric generated by `summary_groups` that has not already appeared. Thus, `dataset_abbrs` filters and orders displayed results; it does not define groups or formulas and cannot rename metrics.

For example, if two subsets have `accuracy` values of 60 and 90 and sample counts of 100 and 200:

```text
naive_average    = (60 + 90) / 2 = 75
weighted_average = (60 × 100 + 90 × 200) / (100 + 200) = 80
```

If any `[abbr, metric]` in `subsets` cannot be found, the entire group is marked `error: missing metrics`; the Summarizer does not calculate a partial score from the remaining subsets.

### Other Summary Group Fields

As in the example, prefer `[abbr, metric]` pairs in `subsets` to select every input metric explicitly. Other fields and forms behave as follows:

| Field                           | Meaning                                                                                                                                                                   |
| ------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `name`                          | The abbreviation of the newly generated group.                                                                                                                            |
| `subsets`                       | Aggregation inputs. Prefer `[Dataset abbr, metric]` pairs. Bare Dataset-abbreviation strings are also supported, but the two forms cannot be mixed.                       |
| `metric`                        | The output metric name, not an input metric name. Input metrics still come from pairs in `subsets`. When omitted, the name is derived from the aggregation method, such as `naive_average` or `weighted_average`. |
| `weights`                       | `{Dataset abbr: weight}`; enables a weighted average. With pair inputs, keys may also use the `Dataset abbr@metric` form.                                                 |
| `std` / `sum` / `harmonic_mean` | Calculate the population standard deviation, sum, or harmonic mean, respectively. To avoid interactions between aggregation methods, select only one for a group and do not combine it with `weights`. |
| `transforms`                    | `{Dataset abbr: 'expression'}`; transforms the corresponding input before aggregation, with `x` representing the original value. This can align metric directions or scales, but it is executed through `eval` and must be used only in trusted configurations. |

For example, PluginEval uses top-level `metric` to name the aggregate output; the two actual input metrics remain in `subsets`:

```python
dict(
    name='plugin_eval-instruct_v1',
    metric='format_metric',
    subsets=[
        ['plugin_eval-instruct_v1', 'string_format_metric'],
        ['plugin_eval-instruct_v1', 'json_format_metric'],
    ],
)
```

This configuration takes the arithmetic mean of the two inputs and stores it as `plugin_eval-instruct_v1/format_metric`.

When every item in `subsets` is a string, `DefaultSummarizer` aggregates every metric shared by all subsets. It also generates one default aggregate metric whose inputs are the first metrics of the individual subsets after sorting by [`METRIC_WHITELIST`](https://github.com/open-compass/opencompass/blob/main/opencompass/summarizers/default.py#L19). Different subsets may have different first metrics, so use this shorthand only when their primary metric conventions are known to match.

### When to Write a Custom Summarizer Class

Add a Summarizer only when the default table cannot express the grouping or aggregation convention. Subclass `DefaultSummarizer` and override `summarize(output_path, time_str)`; repository examples include [`CircularSummarizer`](https://github.com/open-compass/opencompass/blob/main/opencompass/summarizers/circular.py#L11) and [`MultiFacetedSummarizer`](https://github.com/open-compass/opencompass/blob/main/opencompass/summarizers/multi_faceted.py#L14). Set `type=MySummarizer` in configuration. An aggregate score must document:

- Whether it is a macro average or sample-count weighted.
- How missing subsets are handled.
- Whether metric direction and scaling agree.
- How repeated runs are aggregated.

Prefer expressing groups through configuration under `opencompass/configs/summarizers/`. Implement and register a class only for genuinely new behavior. Validate it against a small fixed result set rather than depending on full model inference.
