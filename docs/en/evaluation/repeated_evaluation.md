# Repeated Evaluation, Repeated Sampling, and Stability

A dataset containing 100 questions does not make the framework evaluate it repeatedly. By default, each Dataset runs once. Multiple predictions are produced only through `n` in the configuration, CLI `--dataset-num-runs`, or a dedicated repeated-sampling configuration.

## Setting the Repeat Count in a Configuration

Set `n` directly at the top level of the dataset dict to declare the number of independent runs per sample. A pass@k-style metric normally also sets `k`. The repository's repeated-sampling configuration for humaneval-plus follows this pattern:

```python
humaneval_plus_datasets = [
    dict(
        abbr='humaneval_plus',
        type=HumanevalDataset,
        path='opencompass/humaneval',
        reader_cfg=humaneval_plus_reader_cfg,
        infer_cfg=humaneval_plus_infer_cfg,
        eval_cfg=humaneval_plus_eval_cfg,
        n=5,  # Independently run each sample 5 times (default: 1)
        k=3,  # pass@k level passed to the evaluator; meaningful only when supported
    )
]
```

The two fields mean:

- `n`: repeat count. The evaluation task passes `n` together with predictions to `evaluator.evaluate(k, n, ...)`, and the evaluator interprets it. The default `BaseEvaluator` implementation scores each of the n prediction sets and averages them (when `n>1`, metric names receive a `(5 runs average)` suffix), then groups multiple results for the same sample to compute cross-run metrics such as pass@k and multi-run consistency.
- `k`: pass@k level forwarded to the evaluator, as either one integer or a list such as `k=[1, 10, 100]`. Its exact meaning is determined by the Evaluator.

Setting `n` has two prerequisites:

1. **Inference must actually produce n times as many predictions.** `n` takes effect only during evaluation and does not automatically make a model sample repeatedly. If the model supports multiple replies, set `num_return_sequences=n` in `generation_kwargs`; otherwise, use the dataset loader's `num_repeats` to copy each sample n times (supported by datasets such as humaneval and apps). See the pass@k section of [Code Evaluation](code_eval.md) for complete configurations of both approaches.
2. **The evaluator must understand `n` / `k`**, for example `HumanEvalPlusEvaluator` or `MBPPPassKEvaluator`. Even if an ordinary accuracy evaluator accepts `n`, its output is only a multi-run average and may not be the intended convention.

## Overriding in Bulk from the Command Line

```bash
opencompass my_eval.py --dataset-num-runs 5
```

This is equivalent to overriding both `n` and `k` to 5 in every dataset configuration and is convenient for temporary comparisons. A formal evaluation with a fixed protocol should write `n` / `k` directly in the configuration. This option requires every dataset to have already defined `n`, otherwise it raises an error; do not assume that all metrics output the same pass@k convention.

## When Repetition Is Needed

- Generation uses nonzero temperature or another stochastic sampling strategy.
- An API service has uncontrollable variability.
- A metric such as code pass@k or G-Pass@k explicitly requires multiple candidates.
- Prompt or Judge stability is under study.

With greedy decoding and a deterministic backend, repetition usually adds cost only. Report at least the run count, random arguments, score of each run, mean, and dispersion.

## Repetition Analysis Within a Reply

For a formal run, add:

```bash
opencompass my_eval.py --analysis-repeat
```

You can also analyze an existing timestamp directory:

```bash
python tools/analyze_repeat.py outputs/my_eval/20260903_120000 \
    --model model-abbr \
    --tokenizer gpt-4o
```

The analysis report is written under `summary/` and mainly contains:

- sample counts and mean / p75 / p90 token lengths for each benchmark, plus the length threshold used to select long replies for repetition analysis;
- `repeat_pattern`: periodic repeated fragments, repeat counts, and repeated-fragment ratios, useful for locating loops inside replies;
- `gzip_high_compression`: samples with abnormally high gzip compression ratios, useful for finding large repeated or highly templated outputs;
- `missing_prediction_files`: model / dataset combinations present in the configuration but missing prediction files;
- `abnormal_samples`: the abnormal sample list, including model, benchmark, sample ID, prediction path, triggering metrics, and the original prediction.

This tool detects abnormal repetition patterns within replies; it is not a consistency analysis across repeated evaluations. The tokenizer affects repeated-fragment statistics and should be recorded.
