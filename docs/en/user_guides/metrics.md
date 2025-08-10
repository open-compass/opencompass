# Metric Calculation

In the evaluation phase, we typically select the corresponding evaluation metric strategy based on the characteristics of the dataset itself. The main criterion is the **type of standard answer**, generally including the following types:

- **Choice**: Common in classification tasks, judgment questions, and multiple-choice questions. Currently, this type of question dataset occupies the largest proportion, with datasets such as MMLU, CEval, etc. Accuracy is usually used as the evaluation standard-- `ACCEvaluator`.
- **Phrase**: Common in Q&A and reading comprehension tasks. This type of dataset mainly includes CLUE_CMRC, CLUE_DRCD, DROP datasets, etc. Matching rate is usually used as the evaluation standard--`EMEvaluator`.
- **Sentence**: Common in translation and generating pseudocode/command-line tasks, mainly including Flores, Summscreen, Govrepcrs, Iwdlt2017 datasets, etc. BLEU (Bilingual Evaluation Understudy) is usually used as the evaluation standard--`BleuEvaluator`.
- **Paragraph**: Common in text summary generation tasks, commonly used datasets mainly include Lcsts, TruthfulQA, Xsum datasets, etc. ROUGE (Recall-Oriented Understudy for Gisting Evaluation) is usually used as the evaluation standard--`RougeEvaluator`.
- **Code**: Common in code generation tasks, commonly used datasets mainly include Humaneval, MBPP datasets, etc. Execution pass rate and `pass@k` are usually used as the evaluation standard. At present, Opencompass supports `MBPPEvaluator` and `HumanEvalEvaluator`.

There is also a type of **scoring-type** evaluation task without standard answers, such as judging whether the output of a model is toxic, which can directly use the related API service for scoring. At present, it supports `ToxicEvaluator`, and currently, the realtoxicityprompts dataset uses this evaluation method.

## Supported Evaluation Metrics

Currently, in OpenCompass, commonly used Evaluators are mainly located in the [`opencompass/openicl/icl_evaluator`](https://github.com/open-compass/opencompass/tree/main/opencompass/openicl/icl_evaluator) folder. There are also some dataset-specific indicators that are placed in parts of [`opencompass/datasets`](https://github.com/open-compass/opencompass/tree/main/opencompass/datasets). Below is a summary:

| Evaluation Strategy   | Evaluation Metrics   | Common Postprocessing Method | Datasets                                                             |
| --------------------- | -------------------- | ---------------------------- | -------------------------------------------------------------------- |
| `ACCEvaluator`        | Accuracy             | `first_capital_postprocess`  | agieval, ARC, bbh, mmlu, ceval, commonsenseqa, crowspairs, hellaswag |
| `EMEvaluator`         | Match Rate           | None, dataset-specific       | drop, CLUE_CMRC, CLUE_DRCD                                           |
| `BleuEvaluator`       | BLEU                 | None, `flores`               | flores, iwslt2017, summscreen, govrepcrs                             |
| `RougeEvaluator`      | ROUGE                | None, dataset-specific       | truthfulqa, Xsum, XLSum                                              |
| `JiebaRougeEvaluator` | ROUGE                | None, dataset-specific       | lcsts                                                                |
| `HumanEvalEvaluator`  | pass@k               | `humaneval_postprocess`      | humaneval_postprocess                                                |
| `MBPPEvaluator`       | Execution Pass Rate  | None                         | mbpp                                                                 |
| `ToxicEvaluator`      | PerspectiveAPI       | None                         | realtoxicityprompts                                                  |
| `AGIEvalEvaluator`    | Accuracy             | None                         | agieval                                                              |
| `AUCROCEvaluator`     | AUC-ROC              | None                         | jigsawmultilingual, civilcomments                                    |
| `MATHEvaluator`       | Accuracy             | `math_postprocess`           | math                                                                 |
| `MccEvaluator`        | Matthews Correlation | None                         | --                                                                   |
| `SquadEvaluator`      | F1-scores            | None                         | --                                                                   |

## How to Configure

The evaluation standard configuration is generally placed in the dataset configuration file, and the final xxdataset_eval_cfg will be passed to `dataset.infer_cfg` as an instantiation parameter.

Below is the definition of `govrepcrs_eval_cfg`, and you can refer to [configs/datasets/govrepcrs](https://github.com/open-compass/opencompass/tree/main/configs/datasets/govrepcrs).

```python
from opencompass.openicl.icl_evaluator import BleuEvaluator
from opencompass.datasets import GovRepcrsDataset
from opencompass.utils.text_postprocessors import general_cn_postprocess

govrepcrs_reader_cfg = dict(.......)
govrepcrs_infer_cfg = dict(.......)

# Configuration of evaluation metrics
govrepcrs_eval_cfg = dict(
    evaluator=dict(type=BleuEvaluator),            # Use the common translator evaluator BleuEvaluator
    pred_role='BOT',                               # Accept 'BOT' role output
    pred_postprocessor=dict(type=general_cn_postprocess),      # Postprocessing of prediction results
    dataset_postprocessor=dict(type=general_cn_postprocess))   # Postprocessing of dataset standard answers

govrepcrs_datasets = [
    dict(
        type=GovRepcrsDataset,                 # Dataset class name
        path='./data/govrep/',                 # Dataset path
        abbr='GovRepcrs',                      # Dataset alias
        reader_cfg=govrepcrs_reader_cfg,       # Dataset reading configuration file, configure its reading split, column, etc.
        infer_cfg=govrepcrs_infer_cfg,         # Dataset inference configuration file, mainly related to prompt
        eval_cfg=govrepcrs_eval_cfg)           # Dataset result evaluation configuration file, evaluation standard, and preprocessing and postprocessing.
]
```
