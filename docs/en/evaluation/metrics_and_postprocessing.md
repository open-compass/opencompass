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

Not every Dataset needs both postprocessors. If reference answers are already normalized labels, `dataset_postprocessor` can be omitted.

## Choosing a Metric

- Multiple choice and classification: Accuracy.
- Short answer: Exact Match, F1, or task-specific equivalence.
- Translation: text-generation metrics such as BLEU.
- Summarization: overlap metrics such as ROUGE.
- Mathematics: numeric, expression, or symbolic equivalence.
- Code: isolated execution pass rate and pass@k.
- Open-ended answers: rule-based scoring, a specialized model, or LLM Judge.

Metrics with the same name can still differ in case handling, whitespace, punctuation, multiple references, and averaging. When publishing results, record the concrete Evaluator class and postprocessors, not only the metric name.

## Extraction Failures

An answer-extraction failure should normally count as an error rather than cause the sample to be discarded. Add `--dump-extract-rate` at runtime and inspect the per-sample evaluation details, which are enabled by default:

```bash
opencompass my_eval.py --dump-extract-rate
```

After changing extraction rules, reuse existing predictions and rerun scoring only:

```bash
opencompass my_eval.py --work-dir outputs/my_eval \
    --reuse <timestamp> --mode eval
```

## Validating a New Rule

At minimum, cover correctly formatted answers, explanatory text, multiple candidate answers, empty replies, truncated replies, Unicode/full-width and half-width differences, and adversarial formatting. A more permissive postprocessor creates more false positives; a stricter one creates more false negatives for semantically correct answers. Review rules and test samples together.
