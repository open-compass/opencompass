# Understanding Outputs and Result Summaries

By default, OpenCompass stores each experiment under `<work_dir>/<timestamp>/` with the following structure:

```text
outputs/my_eval/<timestamp>/
├── configs/       # Snapshot of the effective configuration for this run
├── logs/
│   ├── infer/     # Inference task logs
│   └── eval/      # Evaluation task logs
├── predictions/   # Per-sample outputs produced by the Inferencer
├── results/       # Metrics and details produced by the Evaluator
└── summary/       # Summary files produced by the Summarizer
```

Retain the effective configuration and the artifacts from every stage together. When `--reuse` is used later, OpenCompass also relies on these existing files to determine which tasks can be skipped.

## Configuration Snapshots and Logs

`configs/` contains a snapshot produced after OpenCompass parses the configuration and applies command-line overrides; it is not merely a copy of the original configuration file. When investigating differences between results, compare the model, dataset, task, and Summarizer configurations in this directory first.

Without `--debug`, each task writes its standard output and standard error to an `.out` file under `logs/infer/` or `logs/eval/`. Paths are organized by model and dataset abbreviation. With `--debug`, tasks run in the current process and logs are primarily displayed in the terminal, so corresponding per-task log files may not be created.

## Predictions: Per-sample Inference Outputs

Prediction files are written to `predictions/<model abbr>/<dataset abbr>.json`. For example:

```text
predictions/gpt-6-astra-response/demo_gsm8k.json
```

Each file uses sample indices as keys and is the primary source for inspecting model inputs and raw outputs. A typical file has the following structure:

```json
{
    "0": {
        "origin_prompt": [
            {"role": "user", "content": "A printing press prints 36 pages in 4 minutes. How many pages can it print in 10 minutes?"}
        ],
        "prediction": "36 / 4 = 9 pages per minute, and 10 * 9 = 90. #### 90",
        "gold": "Calculation... #### 90"
    },
    "1": {
        "origin_prompt": [
            {"role": "user", "content": "A garment originally costs 120 yuan. What is its price after a 20% discount?"}
        ],
        "prediction": "120 * 0.8 = 96, so the price is 96 yuan. #### 96",
        "gold": "Calculation... #### 96"
    }
}
```

- `origin_prompt`: the input sent to the model after processing by the dataset template and model template. Its exact structure depends on the templates and model class in use.
- `prediction`: the model response saved by the Inferencer, before processing by `eval_cfg.pred_postprocessor`.
- `gold`: the original reference answer from the field specified by `reader_cfg.output_column`.

Depending on the task, files may also contain input or output lengths, PPL values, rollout data, or multi-turn results. Refer to the actual prediction file for its precise structure. When scores are abnormal, first verify that the sample is correct, the prompt is complete, and the response has not been truncated; then inspect the scoring stage.

## Results: Evaluation Metrics and Details

Result files are written to `results/<model abbr>/<dataset abbr>.json`. For example:

```text
results/gpt-6-astra-response/demo_gsm8k.json
```

The evaluation stage reads the prediction file and source data, processes predictions and reference answers according to `eval_cfg`, and then invokes the Evaluator to compute metrics. `--dump-eval-details` is enabled by default so that per-sample evaluation details are retained in the result.

```json
{
    "accuracy": 79.6875,
    "details": [
        {"pred": "90", "answer": "90", "correct": true},
        {"pred": "", "answer": "45", "correct": false}
    ]
}
```

Metric names, value ranges, and the structure of `details` are defined by the specific Evaluator. Comparing prediction files with these details can reveal whether an error originates in answer extraction, reference-answer processing, or scoring logic.

## Summary: Aggregating Results

The Summarizer reads metrics from `results/` and controls their display order, metric selection, and grouped calculations. It does not score predictions again. For example:

```python
summarizer = dict(
    dataset_abbrs=[
        'All Results',
        ['demo_gsm8k', 'accuracy'],
    ],
)
```

When `summarizer.type` is omitted, the entry point automatically uses `DefaultSummarizer`. The explicit equivalent is:

```python
from opencompass.summarizers import DefaultSummarizer

summarizer = dict(
    type=DefaultSummarizer,
    dataset_abbrs=[
        'All Results',
        ['demo_gsm8k', 'accuracy'],
    ],
)
```

`dataset_abbrs` controls the rows in the summary table in list order:

- `['demo_gsm8k', 'accuracy']` explicitly selects a dataset abbreviation and metric;
- specifying only a dataset abbreviation displays its highest-priority numeric metric;
- a string such as `All Results` that does not correspond to a dataset or group is displayed as a separator row;
- if `dataset_abbrs` is omitted, all available dataset and group metrics from the current configuration are displayed by default.

If a specified dataset, metric, or result file does not exist, the corresponding cell displays `-`. `dataset_abbrs` controls presentation only; it does not produce results for datasets excluded from the experiment.

To aggregate results from multiple subsets, configure `summary_groups`. For example:

```python
summarizer = dict(
    dataset_abbrs=[
        ['demo_gsm8k', 'accuracy'],
        'reasoning-average',
    ],
    summary_groups=[
        dict(
            name='reasoning-average',
            subsets=[
                ['demo_gsm8k', 'accuracy'],
                ['another_dataset', 'accuracy'],
            ],
            transforms={
                # Assume this subset reports accuracy on a 0-1 scale;
                # convert it to a percentage before aggregation.
                'another_dataset': 'x * 100',
            },
        ),
    ],
)
```

`name` is the group name shown in the summary table, and `subsets` specifies the datasets and metrics included in the aggregation. `DefaultSummarizer` supports the following strategies:

- if no strategy is specified, it computes the arithmetic mean (macro average) under the metric name `naive_average`;
- `weights={'demo_gsm8k': 1, 'another_dataset': 2}` computes the weighted average as `Σ(w·x) / Σw` under the metric name `weighted_average`;
- `sum=True` computes the sum of subset scores;
- `std=True` computes the population standard deviation;
- `harmonic_mean=True` computes the harmonic mean and requires all scores to be greater than zero.

`transforms` applies a separate transformation to each subset score before aggregation. Each key is a subset `abbr`, and each value is an expression in which `x` represents that subset's original score. This supports percentage conversion and more complex expressions such as `'max((3 - x) / 3, 0) * 100'`. Transformations affect only the summary calculation and do not modify the original metrics in `results/`.

If any required subset or metric is missing, the entire group is marked as missing rather than being calculated from the remaining results. For large-scale evaluations, prefer maintained preset configurations under `opencompass/configs/summarizers/`.

After a run, `summary/` contains `.txt`, `.csv`, and `.md` representations of the same table, such as `summary_20260908_141530.csv`. A similar table is also printed in the terminal:

```text
dataset       version    metric    mode    gpt-6-astra-response
------------  ---------  --------  ------  --------------------
demo_gsm8k    1d7fe4     accuracy  gen                    79.69
```

- `dataset`: the dataset or group `abbr`;
- `version`: the first six characters of the prompt-configuration hash calculated from the dataset `infer_cfg`; this is not the dataset release version;
- `metric`: the metric name produced by the Evaluator;
- `mode`: `gen`, `ppl`, `ll`, or `unknown`, as determined from the Inferencer type;
- subsequent columns: model abbreviations and scores, one column per model.
