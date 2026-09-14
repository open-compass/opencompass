# Understanding Outputs and Result Summarization

By default, OpenCompass places each experiment under `<work_dir>/<timestamp>/`. Preserve the effective configuration together with the artifacts from every stage.

```text
<work_dir>/<timestamp>/
├── configs/       # Final configuration snapshot
├── logs/          # Inference and evaluation task logs
├── predictions/   # Per-sample model output
├── results/       # Metrics and details produced by Evaluator
└── summary/       # Tables and aggregate files produced by Summarizer
```

## Predictions

Prediction files are stored at `predictions/<model abbr>/<dataset abbr>.json`, keyed by sample index, and are the primary evidence for troubleshooting. Inspect the final model input, raw reply, processed prediction, sample index, and error fields together. When a score is anomalous, do not adjust the Summarizer first; confirm that samples loaded correctly, prompts are complete, and replies were not truncated.

```json
{
    "0": {
        "origin_prompt": [
            {"role": "user", "content": "A printing press prints 36 pages in 4 minutes... How many pages can it print in 10 minutes?"}
        ],
        "prediction": "36 / 4 = 9 pages/minute, 10 * 9 = 90. #### 90",
        "gold": "90"
    },
    "1": {
        "origin_prompt": [
            {"role": "user", "content": "A garment originally costs 120 and is sold at a 20% discount..."}
        ],
        "prediction": "120 * 0.8 = 96",
        "gold": "96"
    }
}
```

`origin_prompt` is the final message sent to the model (a `role/content` list for conversational inference), `prediction` is the raw model reply, and `gold` is the reference answer.

## Results

Result files are stored at `results/<model abbr>/<dataset abbr>.json`. The Evaluator reads predictions and references and performs answer extraction, normalization, and metric computation. With the default `--dump-eval-details`, results also contain per-sample details; use `--dump-eval-details False` when disk space is limited.

```json
{
    "accuracy": 0.796875,
    "details": {
        "0": {"pred": "90", "answer": "90", "correct": true},
        "1": {"pred": "96", "answer": "96", "correct": true},
        "2": {"pred": "", "answer": "45", "correct": false}
    }
}
```

Fields in each `details` record vary by Evaluator; the example is illustrative. Comparing details reveals whether an answer-extraction failure or an incorrect model answer caused an error.

When only answer extraction or the Evaluator changes, predictions can normally be reused with `--reuse <timestamp> --mode eval`.

## Summary

A Summarizer controls grouping, order, aliases, and aggregate-metric presentation. The default summarizer prints a basic table directly:

```python
from opencompass.summarizers import DefaultSummarizer

summarizer = dict(type=DefaultSummarizer)
```

After the run, `summary/` contains `.txt`, `.csv`, and `.md` versions of the same table, for example `summary_20260908_141530.csv`. The terminal output looks like:

```text
dataset    version    metric    mode    qwen3.5-35b-a3b-vllm
---------  ---------  --------  ------  ---------------------
gsm8k      1d7fe4     accuracy  gen           79.69
```

`version` comes from the dataset configuration's version identifier. In a multi-model evaluation, each model occupies one column.

Large leaderboards generally inherit a specialized configuration under `opencompass/configs/summarizers/`. An aggregate score may use a macro average, sample-count weighting, or custom grouping.

When only the Summarizer changes, rerun aggregation with `--reuse <timestamp> --mode viz`; inference and scoring do not need to run again.
