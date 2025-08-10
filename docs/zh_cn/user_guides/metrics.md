# 评估指标

在评测阶段，我们一般以数据集本身的特性来选取对应的评估策略，最主要的依据为**标准答案的类型**，一般以下几种类型：

- **选项**：常见于分类任务，判断题以及选择题，目前这类问题的数据集占比最大，有 MMLU, CEval 数据集等等，评估标准一般使用准确率--`ACCEvaluator`。
- **短语**：常见于问答以及阅读理解任务，这类数据集主要包括 CLUE_CMRC, CLUE_DRCD, DROP 数据集等等，评估标准一般使用匹配率--`EMEvaluator`。
- **句子**：常见于翻译以及生成伪代码、命令行任务中，主要包括 Flores, Summscreen, Govrepcrs, Iwdlt2017 数据集等等，评估标准一般使用 BLEU(Bilingual Evaluation Understudy)--`BleuEvaluator`。
- **段落**：常见于文本摘要生成的任务，常用的数据集主要包括 Lcsts, TruthfulQA, Xsum 数据集等等，评估标准一般使用 ROUGE（Recall-Oriented Understudy for Gisting Evaluation）--`RougeEvaluator`。
- **代码**：常见于代码生成的任务，常用的数据集主要包括 Humaneval，MBPP 数据集等等，评估标准一般使用执行通过率以及 `pass@k`，目前 Opencompass 支持的有`MBPPEvaluator`、`HumanEvalEvaluator`。

还有一类**打分类型**评测任务没有标准答案，比如评判一个模型的输出是否存在有毒，可以直接使用相关 API 服务进行打分，目前支持的有 `ToxicEvaluator`，目前有 realtoxicityprompts 数据集使用此评测方式。

## 已支持评估指标

目前 OpenCompass 中，常用的 Evaluator 主要放在 [`opencompass/openicl/icl_evaluator`](https://github.com/open-compass/opencompass/tree/main/opencompass/openicl/icl_evaluator)文件夹下， 还有部分数据集特有指标的放在 [`opencompass/datasets`](https://github.com/open-compass/opencompass/tree/main/opencompass/datasets) 的部分文件中。以下是汇总：

| 评估指标              | 评估策略             | 常用后处理方式              | 数据集                                                               |
| --------------------- | -------------------- | --------------------------- | -------------------------------------------------------------------- |
| `ACCEvaluator`        | 正确率               | `first_capital_postprocess` | agieval, ARC, bbh, mmlu, ceval, commonsenseqa, crowspairs, hellaswag |
| `EMEvaluator`         | 匹配率               | None, dataset_specification | drop, CLUE_CMRC, CLUE_DRCD                                           |
| `BleuEvaluator`       | BLEU                 | None, `flores`              | flores, iwslt2017, summscreen, govrepcrs                             |
| `RougeEvaluator`      | ROUGE                | None, dataset_specification | truthfulqa, Xsum, XLSum                                              |
| `JiebaRougeEvaluator` | ROUGE                | None, dataset_specification | lcsts                                                                |
| `HumanEvalEvaluator`  | pass@k               | `humaneval_postprocess`     | humaneval_postprocess                                                |
| `MBPPEvaluator`       | 执行通过率           | None                        | mbpp                                                                 |
| `ToxicEvaluator`      | PerspectiveAPI       | None                        | realtoxicityprompts                                                  |
| `AGIEvalEvaluator`    | 正确率               | None                        | agieval                                                              |
| `AUCROCEvaluator`     | AUC-ROC              | None                        | jigsawmultilingual, civilcomments                                    |
| `MATHEvaluator`       | 正确率               | `math_postprocess`          | math                                                                 |
| `MccEvaluator`        | Matthews Correlation | None                        | --                                                                   |
| `SquadEvaluator`      | F1-scores            | None                        | --                                                                   |

## 如何配置

评估标准配置一般放在数据集配置文件中，最终的 xxdataset_eval_cfg 会传给 `dataset.infer_cfg` 作为实例化的一个参数。

下面是 `govrepcrs_eval_cfg` 的定义， 具体可查看 [configs/datasets/govrepcrs](https://github.com/open-compass/opencompass/tree/main/configs/datasets/govrepcrs)。

```python
from opencompass.openicl.icl_evaluator import BleuEvaluator
from opencompass.datasets import GovRepcrsDataset
from opencompass.utils.text_postprocessors import general_cn_postprocess

govrepcrs_reader_cfg = dict(.......)
govrepcrs_infer_cfg = dict(.......)

# 评估指标的配置
govrepcrs_eval_cfg = dict(
    evaluator=dict(type=BleuEvaluator),            # 使用常用翻译的评估器BleuEvaluator
    pred_role='BOT',                               # 接受'BOT' 角色的输出
    pred_postprocessor=dict(type=general_cn_postprocess),      # 预测结果的后处理
    dataset_postprocessor=dict(type=general_cn_postprocess))   # 数据集标准答案的后处理

govrepcrs_datasets = [
    dict(
        type=GovRepcrsDataset,                 # 数据集类名
        path='./data/govrep/',                 # 数据集路径
        abbr='GovRepcrs',                      # 数据集别名
        reader_cfg=govrepcrs_reader_cfg,       # 数据集读取配置文件，配置其读取的split，列等
        infer_cfg=govrepcrs_infer_cfg,         # 数据集推理的配置文件，主要 prompt 相关
        eval_cfg=govrepcrs_eval_cfg)           # 数据集结果的评估配置文件，评估标准以及前后处理。
]
```
