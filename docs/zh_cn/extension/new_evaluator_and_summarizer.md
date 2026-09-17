# 新增评测器、后处理器与汇总器

先确定要扩展的层次：后处理器把文本转成规范答案，Evaluator 对预测和参考答案计算指标，Summarizer 组织多个 Dataset 的已有指标。

## 后处理器

后处理器应是可重复、无副作用的转换，并明确处理空回复、多个答案、格式错误和异常值。普通 Dataset 配置应优先将后处理器放在 `eval_cfg` 顶层：`pred_postprocessor` 在进入 Evaluator 前处理模型预测，`dataset_postprocessor` 处理测试集中的参考答案。

```python
eval_cfg = dict(
    pred_postprocessor=dict(type=my_pred_postprocess),
    dataset_postprocessor=dict(type=my_reference_postprocess),
    evaluator=dict(type=MyEvaluator),
)
```

`pred_postprocessor` 也可以配置在 `eval_cfg.evaluator` 内部。此时 Evaluator 的构造函数必须接收该参数并传给 `BaseEvaluator`，基类会在每份重复运行调用 `score()` 前应用它：

```python
eval_cfg = dict(
    evaluator=dict(
        type=MyEvaluator,
        pred_postprocessor=dict(type='my_pred_postprocess'),
    ),
)
```

这两种配置位于不同的执行阶段，并且会依次生效，而不是互相覆盖。同一个后处理器不要同时配置在两处，否则预测会被重复处理。除非后处理逻辑需要与特定 Evaluator 绑定，否则推荐使用第一种 Dataset 级配置。

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

### score() 参数来源

`score()` 的参数并不限于 `predictions` 和 `references`，但必须是评测任务能够提供的字段。当前 `OpenICLEvalTask` 会先收集预测文件中的字段，再补充或覆盖 `predictions`、`references`、`test_set` 和 `origin_prompt`，最后按照 `score()` 的函数签名选取同名字段传入。

不要在 `score()` 中使用 `**kwargs`。当前实现会将它识别为名为 `kwargs` 的参数，但评测任务没有提供这个字段，因而会在组装参数时出错。也不要直接声明只存在于 Dataset 中的列名；评测任务不会自动展开 Dataset 的任意列，需要题目、选项、测试用例或其他元信息时，应声明 `test_set` 并从中读取。

常用参数如下，其中前四个由评测任务补充或覆盖，其余参数只有在 inferencer 将同名字段写入预测文件时才可使用：

| 参数               | 含义                                                                                                                                    |
| ------------------ | --------------------------------------------------------------------------------------------------------------------------------------- |
| `predictions`      | 模型预测结果列表。生成式任务中，这是经过已配置的模型级、Dataset 级和 Evaluator 级后处理后的结果；一次返回多条候选时也可能是列表的列表。 |
| `references`       | 参考答案列表，来自 `reader_cfg.output_column` 指定的测试集列；未配置 `output_column` 时为 `None`。                                      |
| `test_set`         | 当前测试集的 `datasets.Dataset` 对象，已经过可选的 `dataset_postprocessor` 处理。Dataset 中未单独传入的字段应通过该对象读取。           |
| `origin_prompt`    | 推理阶段写入预测文件的原始 prompt 或 message；预测文件没有该字段时，评测任务会补成与预测数量等长的 `None` 列表。                        |
| `gold`             | 预测文件中的标准答案字段，通常由部分 inferencer 写入；它不等同于固定提供的 `references`，只有预测文件包含该字段时才能声明。             |
| `steps`            | 预测文件或自定义推理流程写入的中间步骤信息，常用于同时评估最终答案和推理步骤。                                                          |
| `res_length`       | 生成结果的长度统计，通常在开启结果长度 dump 时由生成式 inferencer 写入。                                                                |
| `all_input_length` | 输入 prompt 或 message 的总长度统计，通常与 `res_length` 一起用于分析输入输出长度。                                                     |
| `ppl`              | PPL/困惑度相关推理结果，通常由 PPL 类 inferencer 或自定义预测文件提供。                                                                 |
| `token_len`        | 与 `ppl` 配套的 token 数量，用于按 token 数归一化 PPL 等指标。                                                                          |
| `loss`             | 损失值列表，常用于 BPC 等基于 loss 的指标。                                                                                             |
| `total_chr_num`    | 与 `loss` 配套的字符数量，常用于计算 bits per character。                                                                               |
| `mink`             | Min-K 概率类统计值，供对应的 Min-K evaluator 使用。                                                                                     |
| `prompt`           | 预测文件中的 prompt 字段，部分 PPL/条件概率类推理流程会记录该字段。                                                                     |
| `choices`          | 条件概率类推理流程写入的候选项列表。                                                                                                    |
| `pred_label`       | 条件概率类推理流程根据分数选出的预测标签。                                                                                              |

除上述字段外，如果自定义 inferencer 在预测文件中写入了其他键，也可以在 `score()` 中声明同名参数。由于这些字段依赖具体推理流程，使用前应先确认预测文件确实包含它们。

参数组装完成后，评测任务不会直接调用 `score()`，而是调用基类的 `evaluate(k, n, original_dataset, **score_kwargs)`。基类依次完成：

1. 按 `n` 将同一数据集的多轮推理结果划分为独立批次，每批包含一轮完整数据集对应的参数；
2. 对每份预测应用 Evaluator 内部的 `pred_postprocessor`，再调用 `score()`；
3. 汇总各份的数值指标，`n > 1` 时计算均值并在指标名后追加 `(n runs average)`；但当前实现只有在 `score()` 每次调用都返回完整且非空的 `details` 时才会返回该聚合结果，否则最终返回最后一份的评分结果；
4. 弹出并聚合 `details`，跨份按样本分组。

### score() 方法的格式

- 返回 `dict[指标名, 数值]`，数值必须是 `int` / `float`——汇总阶段只保留数值型结果，其他类型会被静默丢弃；返回含 `'error'` 键时整条结果被跳过并在日志中记录；
- 指标名要稳定、可读。常见名称（`accuracy`、`exact_match`、`f1`、`rouge1` 等）在汇总表中排序靠前，冷门名称排在白名单之后；
- `'details'`（`list[dict]`）在接口形式上可选，但当前实现存在限制：当 `n > 1` 且需要正确返回跨份平均指标时，`score()` 每次调用都必须返回与当前批次样本一一对应的完整且非空 `details`；否则最终只会返回最后一份的评分结果。基类会将 `details` 跨重复运行聚合并写回结果文件，供 `--dump-eval-details` 逐条复核。每条 detail 不必包含正确性字段；只有计算 G-Pass@k、mG-Pass@k 等跨次指标时，才需要提供 `correct` / `is_correct` / `cascade_correct` 布尔字段；
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
