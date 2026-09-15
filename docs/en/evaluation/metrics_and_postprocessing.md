# Metrics, Answer Extraction, and Postprocessing

The evaluation method is determined jointly by the problem format, model capability, and intended use of the result. OpenCompass separates producing predictions from scoring them, so answer extraction or the Evaluator can be changed for the same predictions without requesting the model again.

Generative evaluation obtains replies through `GenInferencer`, then scores them with rules, numeric equivalence, text metrics, or execution results. PPL evaluation compares candidate probabilities through `PPLInferencer` and applies only to model backends that expose token probabilities. Open-ended answers can also use [LLM Judge](llm_judge.md) or [Cascade Evaluation](cascade_evaluator.md). Mathematics, code, subjective, and multimodal tasks require their corresponding specialized evaluation methods.

A raw model reply usually cannot be compared directly with a reference answer. Objective evaluation generally has three steps: extract an answer from the reply, normalize both prediction and reference, and aggregate the metric with an Evaluator.

```text
raw reply → pred_postprocessor → normalized prediction
reference answer → dataset_postprocessor → normalized answer
normalized prediction + normalized answer → Evaluator → metric
```

## Configuring It in a Dataset

```python
eval_cfg = dict(
    pred_role='BOT',
    pred_postprocessor=dict(type=my_pred_postprocess),
    dataset_postprocessor=dict(type=my_reference_postprocess),
    evaluator=dict(type=MyEvaluator),
)
```

### Postprocessor Interface

`my_pred_postprocess` and `my_reference_postprocess` are both called per sample. For `my_pred_postprocess`, the interface should satisfy these requirements:

- The first positional argument receives one model output, usually a `str`. If the prediction is a list of candidates, OpenCompass calls the function on each string in that list.
- Fields in `dict(type=..., key=value)` other than `type` are passed as keyword arguments, so the function can declare extra parameters as needed, for example `def my_pred_postprocess(text: str, option: str = 'ABCD')`.
- The return value should be one normalized prediction, not a batch of predictions. Its type must be compatible with the `evaluator` and the postprocessed reference, commonly an option letter, short string, number, or task-specific structure.
- Extraction failures should return a value the Evaluator can handle explicitly, such as the original text, an empty string, or a configured invalid value. To make `--dump-extract-rate` count failed samples as extraction failures, return an empty value such as `''` or `None`; returning the original text usually keeps it as a normal prediction for scoring. Raise an exception only when the evaluation should stop.

`dataset_postprocessor` has the same interface, but its input comes from the dataset reader's output column, namely the reference answer.

Not every Dataset needs both postprocessors. If reference answers are already normalized labels, `dataset_postprocessor` can be omitted.

### Existing Config Example

Existing configs also include examples that use all four fields together. For example, in [opencompass/configs/datasets/bbh/bbh_new_gen.py](https://github.com/open-compass/opencompass/blob/main/opencompass/configs/datasets/bbh/bbh_new_gen.py#L53-L57), both `pred_postprocessor` and `dataset_postprocessor` use `bbh_mcq_postprocess`, and the normalized outputs are then scored by `BBHEvaluator_mcq`:

```python
bbh_eval_cfg = dict(
    evaluator=dict(type=BBHEvaluator_mcq),
    pred_role='BOT',
    pred_postprocessor=dict(type=bbh_mcq_postprocess),
    dataset_postprocessor=dict(type=bbh_mcq_postprocess))
```

The implementation of `bbh_mcq_postprocess` is in [opencompass/datasets/bbh.py](https://github.com/open-compass/opencompass/blob/main/opencompass/datasets/bbh.py#L32-L44):

```python
@TEXT_POSTPROCESSORS.register_module('bbh-mcq')
def bbh_mcq_postprocess(text: str) -> str:
    ans = text
    ans_line = ans.split('answer is ')
    if len(ans_line) != 1:
        ans = ans_line[1].strip()
    match = re.search(r'\(([A-Z])\)*', ans)
    if match:
        return match.group(1)
    match = re.search(r'([A-Z])', ans)
    if match:
        return match.group(1)
    return ans
```

It first tries to keep the text after `answer is `, then extracts an uppercase option in a form such as `(A)`. If that form is absent, it extracts the first uppercase letter; if nothing matches, it returns the original text.

## Validating Custom Postprocessors

When adding or changing `pred_postprocessor` / `dataset_postprocessor`, at minimum cover correctly formatted answers, explanatory text, multiple candidate answers, empty replies, truncated replies, Unicode/full-width and half-width differences, and adversarial formatting. A more permissive postprocessor creates more false positives; a stricter one creates more false negatives for semantically correct answers. Review rules and test samples together.

## Choosing an Evaluator for a Dataset

In OpenCompass, the evaluation method is usually determined by `eval_cfg.evaluator` in the dataset config. For datasets already supported by OpenCompass, prefer the Evaluator class and postprocessors configured in the corresponding config file; those settings are usually aligned with the dataset's answer format, extraction rules, and aggregation protocol.

Choose or implement an Evaluator manually only when adding a new dataset, changing the scoring protocol for reused predictions, or confirming that the existing config does not match the task requirements. In that case, start from the task output format and decide what format the postprocessors and Evaluator should receive:

- Multiple-choice and classification tasks usually compare normalized options or class labels, with Accuracy as a common aggregation metric.
- Short-answer tasks usually compare normalized short text, using Exact Match, F1, or task-specific equivalence.
- Translation and summarization tasks usually keep more complete generated text, then apply text-generation metrics such as BLEU or ROUGE.
- Mathematics tasks usually extract the final answer first, then apply numeric, expression, or symbolic equivalence.
- Code tasks usually extract executable code first, then compute isolated execution pass rate or pass@k.
- Open-ended answers usually require rule-based scoring, a specialized model, or LLM Judge.

Metrics with the same name can still differ in case handling, whitespace, punctuation, multiple references, and averaging. If the default config is changed, record the concrete Dataset config, Evaluator class, postprocessors, and key parameters, not only the metric name.

## Extraction Failures

An answer-extraction failure should normally count as an error rather than cause the sample to be discarded. Add `--dump-extract-rate` at runtime and inspect the per-sample evaluation details, which are enabled by default:

```bash
opencompass my_eval.py --dump-extract-rate
```

The implementation of `--dump-extract-rate` is in [opencompass/tasks/openicl_eval.py](https://github.com/open-compass/opencompass/blob/main/opencompass/tasks/openicl_eval.py#L437-L454). It reads the `predictions` field in per-sample details; if that field is empty, such as `''`, `None`, or an empty list, the sample is counted as an extraction failure. For generative evaluation, the `predictions` field in details usually comes from `details[i]['pred']` returned by the Evaluator, as shown in [opencompass/tasks/openicl_eval.py](https://github.com/open-compass/opencompass/blob/main/opencompass/tasks/openicl_eval.py#L499-L505).

Therefore, whether a failed extraction is counted depends on the convention between the postprocessor and the Evaluator. If the postprocessor returns `''` or `None` on failure, the sample is included in the `extract_rate` failure count. If it returns the original text, as `bbh_mcq_postprocess` does above, a non-empty original text is not treated as an extraction failure by `extract_rate`; it is passed to the Evaluator as a normal prediction.

After changing extraction rules, reuse existing predictions and rerun scoring only:

```bash
opencompass my_eval.py --work-dir outputs/my_eval \
    --reuse <timestamp> --mode eval
```
