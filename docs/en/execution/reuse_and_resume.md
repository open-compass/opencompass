# Task Recovery, Reuse, and Evaluation-Only Reruns

OpenCompass stores predictions, scores, and summaries separately, so completed stages can be reused. Reuse requires the work directory, timestamp, model and dataset abbreviations, and partitioning scheme to remain consistent.

## Reusing the Most Recent Run

```bash
opencompass my_eval.py --work-dir outputs/my_eval --reuse
```

`--reuse` without a value selects the latest timestamp directory by name under that work directory. In production, explicitly specifying the timestamp is preferable:

```bash
opencompass my_eval.py \
    --work-dir outputs/my_eval \
    --reuse 20260903_120000
```

## Running Stages Separately

```bash
# Generate predictions only
opencompass my_eval.py --work-dir outputs/my_eval --mode infer

# Re-evaluate existing predictions
opencompass my_eval.py --work-dir outputs/my_eval \
    --reuse 20260903_120000 --mode eval

# Re-summarize existing results
opencompass my_eval.py --work-dir outputs/my_eval \
    --reuse 20260903_120000 --mode viz
```

The `eval` and `viz` modes must identify an existing experiment through `--reuse`, or use the result-station loading mechanism.

## Safe Reuse Rules

- Only the Summarizer changed: results can usually be reused; run `viz`.
- Only answer extraction or the Evaluator changed: predictions can be reused; run `eval`.
- The prompt, few-shot examples, model, or generation arguments changed: inference must be rerun.
- The shard count, abbreviation, or sample range changed: treat it as a new experiment unless compatibility has been verified file by file.

Do not assume content is valid merely because a target file exists. After recovery, inspect failure logs, prediction and result counts, and the configuration snapshot.
