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

1. **Inference must produce one prediction for every repeated row.** The current `BaseDataset` loader duplicates each split `n` times before inference. For this route, leave `num_return_sequences=1` and do not also set loader-specific `num_repeats`. Some code evaluators use packed candidate lists instead; see [Code Evaluation](code_eval.md) for those configurations, and avoid combining the two repetition mechanisms.
2. **The evaluator must understand `n` / `k`**, for example `HumanEvalPlusEvaluator` or `MBPPPassKEvaluator`. Even if an ordinary accuracy evaluator accepts `n`, its output is only a multi-run average and may not be the intended convention.

## Bayes@N posterior mean and standard deviation

Set `eval_cfg.bayes_at_n=True` to estimate the average success probability over the questions in one dataset. This option works with evaluators that use `BaseEvaluator.evaluate` and return one binary correctness detail per prediction, such as `EMEvaluator`. The field can be `correct`, `is_correct`, or `cascade_correct`. Evaluators that override `evaluate` without this option, omit details, or return partial credit are unsupported.

For example, save these two questions as `questions.jsonl`:

```json
{"question": "What is two plus two?", "answer": ["4", "four"]}
{"question": "What is the capital of France?", "answer": ["Paris"]}
```

The following configuration uses the existing model configuration and a local dataset:

```python
from mmengine.config import read_base

with read_base():
    from opencompass.configs.models.qwen2_5.hf_qwen2_5_0_5b_instruct import models

from opencompass.datasets import CustomDataset
from opencompass.openicl.icl_evaluator import EMEvaluator
from opencompass.openicl.icl_inferencer import GenInferencer
from opencompass.openicl.icl_prompt_template import PromptTemplate
from opencompass.openicl.icl_retriever import ZeroRetriever

models[0]['generation_kwargs'] = dict(do_sample=True, temperature=0.7)
datasets = [dict(
    type=CustomDataset,
    abbr='questions',
    path='questions.jsonl',
    n=8,
    k=1,
    reader_cfg=dict(input_columns=['question'], output_column='answer'),
    infer_cfg=dict(
        prompt_template=dict(
            type=PromptTemplate,
            template=dict(round=[dict(role='HUMAN', prompt='{question}')])),
        retriever=dict(type=ZeroRetriever),
        inferencer=dict(type=GenInferencer, max_out_len=64)),
    eval_cfg=dict(evaluator=dict(type=EMEvaluator), bayes_at_n=True),
)]
```

Run it with `opencompass bayes_eval.py` after saving the configuration as `bayes_eval.py`. The current `BaseDataset` loader repeats each split `n` times, in replica order, before inference. For this configuration, generate one completion for each repeated row; leave `num_return_sequences=1` and do not also set `num_repeats`. Use independent stochastic generations under the same sampling and grading settings. Reusing a single completion eight times does not supply eight independent trials. If you select `test_range`, it must retain every replica of the selected questions.

Results contain `Bayes@8` (posterior mean) and `Bayes@8 posterior SD` (posterior standard deviation). Both use percentage points, matching accuracy scores. The same fields are reported under each full subdivision name. They appear in the result JSON and the default per-dataset summary, even when prediction details are not saved. Existing scores are retained, and `k` does not affect these posterior moments. The option also supports `n=1`.

With `s_i` successes in `N=n` trials, question `i` has posterior `Beta(s_i+1, N-s_i+1)`. Its mean is `(s_i+1)/(N+2)` and variance is `(s_i+1)(N-s_i+1)/((N+2)^2(N+3))`. For `M` questions, Bayes@N averages the question means and reports `sqrt(sum(question variances))/M` as posterior SD, then multiplies both by 100. For outcomes `[[0, 0], [1, 1]]`, the result is a mean of `50` and a posterior SD of approximately `13.6931` percentage points. This SD measures uncertainty about the fixed question set's latent mean success probability, rather than variation across runs or across benchmark questions.

Missing predictions, incomplete details, failed graders, nonbinary labels, and duplicate or inconsistent question IDs raise `ValueError`. Each question must occur once in every replica. This integration uses a uniform `Beta(1,1)` prior; categorical rubrics, prior observations, and credible intervals are not included. Cross-dataset `summary_groups` omit these fields because averaging dataset posterior SDs or means does not give a corpus posterior. Explicitly selecting Bayes@N fields in such a group raises an error. Evaluate a combined question set as one dataset to obtain its posterior moments.

The formulas are the binary case of [Bayes@N, Algorithm 1](https://arxiv.org/abs/2510.04265), also available in the [Scorio reference implementation](https://github.com/mohsenhariri/scorio/blob/7278f677c30396d99a0454bcbb08f5a9a53c4efb/scorio/eval/bayes.py). No additional dependency is required.

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
