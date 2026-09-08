# 新增评测器、后处理器与汇总器

先确定要扩展的层次：后处理器把文本转成规范答案，Evaluator 对预测和参考答案计算指标，Summarizer 组织多个 Dataset 的已有指标。

## 后处理器

后处理器应是可重复、无副作用的转换，并明确处理空回复、多个答案、格式错误和异常值。配置通常位于 Dataset 的 `eval_cfg`：

```python
eval_cfg = dict(
    pred_postprocessor=dict(type=my_pred_postprocess),
    dataset_postprocessor=dict(type=my_reference_postprocess),
    evaluator=dict(type=MyEvaluator),
)
```

## Evaluator

### 实现需求

Evaluator 继承 `BaseEvaluator`（`opencompass.openicl.icl_evaluator`）并用 `ICL_EVALUATORS` 注册。**唯一必须实现的方法是 `score()`**；构造函数若接受 `pred_postprocessor` 并透传给 `super().__init__`，框架会在每次评分前自动应用它。最小骨架：

```python
from opencompass.openicl.icl_evaluator import BaseEvaluator
from opencompass.registry import ICL_EVALUATORS


@ICL_EVALUATORS.register_module()
class MyEvaluator(BaseEvaluator):

    def __init__(self, pred_postprocessor=None):
        super().__init__(pred_postprocessor=pred_postprocessor)

    def score(self, predictions, references, test_set=None):
        details = []
        correct = 0
        for pred, ref in zip(predictions, references):
            is_correct = str(pred).strip() == str(ref).strip()
            correct += int(is_correct)
            details.append(
                dict(pred=pred, answer=ref, correct=is_correct))
        return {'accuracy': 100 * correct / len(predictions),
                'details': details}
```

评测任务不直接调用 `score()`，而是经过基类的 `evaluate(k, n, original_dataset, **score_kwargs)`，它依次完成：

1. 按 `score()` 的函数签名组装参数——`predictions`、`references`、`test_set`，以及测试集中与签名参数同名的其他列（需要题目元信息时直接在签名里声明列名即可）；
2. 按 `n` 把预测切成 n 份逐份评分，评分前对每份应用 `pred_postprocessor`；
3. 数值指标跨份取均值，`n > 1` 时指标名追加 `(n runs average)`；
4. 弹出并聚合 `details`，跨份按样本分组。

### score() 方法的格式

- 返回 `dict[指标名, 数值]`，数值必须是 `int` / `float`——汇总阶段只保留数值型结果，其他类型会被静默丢弃；返回含 `'error'` 键时整条结果被跳过并在日志中记录；
- 指标名要稳定、可读。常见名称（`accuracy`、`exact_match`、`f1`、`rouge1` 等）在汇总表中排序靠前，冷门名称排在白名单之后；
- 可选返回 `'details'`（`list[dict]`）：基类会跨重复运行聚合并写回结果文件，供 `--dump-eval-details` 逐条复核。每条 detail 若带 `correct` / `is_correct` / `cascade_correct` 布尔字段，在 `n > 1` 且 `k > 1` 时基类还会自动计算 G-Pass@k、mG-Pass@k 等跨次指标；
- 自定义类不应重写 `evaluate()`，除非确有必要改变重复运行的切分语义。

实现时应返回稳定的指标名称，并针对正常、空输入、解析失败、边界值及多参考答案编写测试。若评测器会访问网络、调用 Judge 或执行代码，必须提供超时、错误记录和隔离策略。

## Summarizer

### 优先用 summary_groups 配置

汇总器配置不写 `type` 时默认使用 `DefaultSummarizer`，它从 `<work_dir>/results/` 收集各模型—数据集的指标文件，计算分组聚合后输出 `<work_dir>/summary/` 下的 txt / csv / md 汇总表。`dataset_abbrs` 控制表格的行顺序，可以混入分组名和空字符串 `''`（空行分隔，便于阅读）。

绝大多数需求——分组、平均、加权——都应该用 `summary_groups` 配置表达，而不是新写 Summarizer 类。

### summary group 的字段

| 字段                            | 类型 | 说明                                                                               |
| ------------------------------- | ---- | ---------------------------------------------------------------------------------- |
| `name`                          | str  | 分组名，汇总表中作为一行出现                                                       |
| `subsets`                       | list | 子集列表，元素为数据集 abbr 字符串或 `[abbr, metric]` 二元组，**两种写法不能混用** |
| `metric`                        | str  | 显式指定参与聚合的指标名                                                           |
| `weights`                       | dict | `{子集 abbr: 权重}`，启用加权平均（见下文）                                        |
| `std` / `sum` / `harmonic_mean` | bool | 分别改为标准差、求和、调和平均                                                     |
| `transforms`                    | dict | `{子集 abbr: '表达式'}`，聚合前对该子集的分数做变换，表达式中的 `x` 代表原值       |

聚合方式的判定顺序：显式 `metric` > `std` / `sum` / `weights` / `harmonic_mean` 对应的方式 > 默认的简单平均（宏平均）。

**任一子集缺少结果时整组直接标记 `error: missing metrics`，不会出部分聚合值**——这保证了分组分数永远基于完整子集，调试时可据此定位缺失的数据集。

### 什么时候可以在 subsets 里直接写数据集名

`subsets` 全部为字符串时，分组会对所有子集的**公共指标**逐个求平均，并额外生成一行聚合指标；这一行取的是**每个子集各自的首个指标**（按 `accuracy` > `exact_match` > … 的白名单优先级排序后的主指标）。因此：

- 各子集的主指标口径**一致**时（例如 mmlu 的 57 个科目全部输出 `accuracy`），直接写数据集名即可，聚合行就是想要的平均分；
- 各子集指标口径**不一致**，或某个子集有多个指标、默认取中的不是你想聚合的那个时，必须用 `[子集 abbr, 指标名]` 二元组显式指定——此时分组只聚合指定的指标，不再做公共指标平均。

`transforms` 可以在聚合前统一量纲（例如把 0–1 的分数乘 100），但它通过 `eval` 执行表达式，只应写在可信配置里。

### 加权平均（weights）

`weights` 为每个子集声明权重，聚合结果行名为 `weighted_average`，计算方式为 `Σ(w·x) / Σw`（权重为 0 的子集跳过，避免其 NaN 分数污染结果）。仓库中的典型用法是让同一批子集以两种口径并排出分：

```python
# opencompass/configs/summarizers/groups/mmlu.py（节选）
mmlu_summary_groups.append({'name': 'mmlu', 'subsets': _mmlu_all})
mmlu_summary_groups.append(
    {'name': 'mmlu-weighted', 'subsets': _mmlu_all, 'weights': _mmlu_weights})
```

`mmlu` 是各科目的宏平均，`mmlu-weighted` 以各科目官方样本数为权重（等价于按样本量加权）；RewardBench 类配置则用官方占比做权重，让总分与官方榜单对齐。`weights` 的键就是子集 abbr，也支持 `子集@指标` 形式。

### 什么时候写自定义 Summarizer 类

只有在默认表格无法表达分组和聚合口径时才新增 Summarizer：继承 `DefaultSummarizer` 并覆盖 `summarize(output_path, time_str)`（仓库内的 `CircularSummarizer`、`MultiFacetedSummarizer` 等为例），配置中写 `type=MySummarizer`。综合分数必须说明：

- 宏平均还是按样本数加权；
- 缺失子集怎样处理；
- 指标方向和缩放是否一致；
- 重复运行如何聚合。

推荐先在 `opencompass/configs/summarizers/` 中用配置表达分组；确需新行为时再实现并注册 Summarizer 类。验收时用一个小型固定结果集验证汇总值，而不是依赖完整模型推理。
