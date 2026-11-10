# Inspecting Predictions and Analyzing Error Cases

## Case Analyzer

```bash
python tools/case_analyzer.py my_eval.py -w outputs/my_eval/<timestamp>
```

It reads existing predictions from `<work_dir>/predictions/`, reloads the Dataset to obtain reference answers, and writes samples for inspection and complete samples under `<work_dir>/case_analysis/bad/` and `<work_dir>/case_analysis/all/`, respectively. It does not read evaluation results from `<work_dir>/results/`.

For PPL evaluation, samples whose predictions differ from their references are written to `bad`. For generation evaluation, all samples are currently written to `bad`, so `bad` represents the complete set of samples awaiting manual inspection rather than samples that an Evaluator has determined to be incorrect.

Before use, confirm that `-w` points to the actual timestamp directory rather than a parent directory containing multiple experiments.

## Merging Sharded Predictions

```bash
python tools/prediction_merger.py my_eval.py \
    -w outputs/my_eval \
    -r <timestamp>
```

Prediction Merger combines shards produced by a Partitioner. `-w` is the work directory containing timestamp directories, while `-r` selects a specific timestamp (`latest` by default). Do not change the Dataset abbreviation, sample range, or partitioning strategy before merging. After merging, check sample IDs for duplicates and omissions.

## Manual Inspection Order

1. Compare against the effective configuration under `configs/`.
2. Inspect failure and retry logs.
3. Compare model input, raw reply, and postprocessed answer.
4. Check empty replies, truncation, parsing failures, and abnormally long output.
5. Only then determine whether the problem comes from the model, prompt, Loader, or Evaluator.
