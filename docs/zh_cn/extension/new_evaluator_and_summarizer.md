# 新增后处理器、评测器与汇总器

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

### 完整例子：InverseIFEval

汇总器配置不写 `type` 时默认使用 `DefaultSummarizer`。它从 `<work_dir>/results/` 读取各个 Dataset 的数值指标，根据 `summary_groups` 生成分组指标，再把表格写到 `<work_dir>/summary/` 下。

下面以仓库中的 InverseIFEval 配置为基础，抽取总分、中文分组和宏平均三组，组成一个可独立理解的完整例子。实际分组定义位于 [`opencompass/configs/summarizers/groups/inverse_ifeval.py`](https://github.com/open-compass/opencompass/blob/main/opencompass/configs/summarizers/groups/inverse_ifeval.py)：

```python
inverse_ifeval_instruction_type_abbrs = [
    'QC', 'ITF', 'CC', 'CCF', 'DIA', 'II', 'MIM', 'CA'
]
inverse_ifeval_language_abbrs = ['zh', 'en']

inverse_ifeval_subsets = [
    f'InverseIFEval_{language}_{instruction_type}'
    for language in inverse_ifeval_language_abbrs
    for instruction_type in inverse_ifeval_instruction_type_abbrs
]

# 每种指令类型的样本数；中英文各占一半。
inverse_ifeval_type_counts = {
    'QC': 90,
    'ITF': 86,
    'CC': 198,
    'CCF': 82,
    'DIA': 186,
    'II': 154,
    'MIM': 108,
    'CA': 108,
}
inverse_ifeval_weights = {
    f'InverseIFEval_{language}_{instruction_type}':
    inverse_ifeval_type_counts[instruction_type] // 2
    for language in inverse_ifeval_language_abbrs
    for instruction_type in inverse_ifeval_instruction_type_abbrs
}

inverse_ifeval_summary_groups = [
    # 按各子集样本数计算总加权平均。
    dict(
        name='InverseIFEval',
        subsets=[[subset, 'accuracy'] for subset in inverse_ifeval_subsets],
        weights=inverse_ifeval_weights,
    ),
    # 只汇总中文子集，并按样本数加权。
    dict(
        name='InverseIFEval_zh',
        subsets=[[
            f'InverseIFEval_zh_{instruction_type}', 'accuracy'
        ] for instruction_type in inverse_ifeval_instruction_type_abbrs],
        weights={
            f'InverseIFEval_zh_{instruction_type}':
            inverse_ifeval_type_counts[instruction_type] // 2
            for instruction_type in inverse_ifeval_instruction_type_abbrs
        },
    ),
    # 所有子集权重相同的宏平均。
    dict(
        name='InverseIFEval_macro',
        subsets=[[subset, 'accuracy'] for subset in inverse_ifeval_subsets],
    ),
]
```

对应的主汇总配置导入上述分组，并指定输出表格的内容和顺序。仓库中的完整版本位于 `opencompass/configs/summarizers/inverse_ifeval.py`：

```python
from mmengine.config import read_base

with read_base():
    from .groups.inverse_ifeval import (inverse_ifeval_subsets,
                                        inverse_ifeval_summary_groups)

summarizer = dict(
    dataset_abbrs=[
        ['InverseIFEval', 'weighted_average'],
        ['InverseIFEval_zh', 'weighted_average'],
        ['InverseIFEval_macro', 'naive_average'],
        *[[subset, 'accuracy'] for subset in inverse_ifeval_subsets],
    ],
    summary_groups=inverse_ifeval_summary_groups,
)
```

这份配置的处理过程如下：

1. `subsets` 中的每个 `[Dataset abbr, metric]` 二元组精确指定一个输入值。例如 `['InverseIFEval_zh_QC', 'accuracy']` 表示读取 `InverseIFEval_zh_QC` 结果文件中的 `accuracy`。这里的二元组用于**选择聚合输入**。
2. `name` 是新生成的分组 abbr。同一个 `name` 下可以产生一个或多个聚合指标。
3. 存在 `weights` 时，结果为 `Σ(w·x) / Σw`，默认指标名是 `weighted_average`。这里权重是各子集的样本数，所以 `InverseIFEval` 是按全部样本计算的加权平均。
4. 不配置 `weights` 或其他聚合方式时，结果为所有输入值的算术平均，默认指标名是 `naive_average`。因此 `InverseIFEval_macro` 是 16 个子集等权的宏平均。
5. `summary_groups` 计算完成后，`dataset_abbrs` 从原始 Dataset 指标和新生成的分组指标中选择要写入汇总表的行。

#### dataset_abbrs 如何控制输出表格

配置了 `dataset_abbrs` 时，列表中的每个元素对应汇总表中的一行，列表顺序就是表格行顺序：

- `[abbr, metric]` 精确选择一行。例如 `['InverseIFEval', 'weighted_average']` 显示 `InverseIFEval` 分组的 `weighted_average`，`['InverseIFEval_zh_QC', 'accuracy']` 显示原始子集的 `accuracy`。这里的二元组用于**选择聚合输出**，不会触发或改变聚合计算。
- 只写 `abbr` 字符串时，显示该 abbr 排序后的首个指标。指标会按照 [`METRIC_WHITELIST`](https://github.com/open-compass/opencompass/blob/main/opencompass/summarizers/default.py#L19) 排序，白名单外的指标排在后面；因此这种写法依赖默认排序，不如二元组明确。
- 写空字符串 `''` 时，由于找不到同名结果，表格会生成一行名称为空、其余单元格为 `-` 的占位行，可作为视觉分隔。
- 如果指定的 abbr 或 metric 不存在，同样会输出一行 `-`，而不是改用其他指标。某个 abbr 和 metric 存在、但某个模型缺少对应结果时，仅该模型的单元格显示 `-`。

以 `['InverseIFEval', 'weighted_average']` 为例，它的查找过程是：

1. `DefaultSummarizer` 先读取原始 Dataset 的结果，再依次处理 `inverse_ifeval_summary_groups`。

2. 它找到 `name='InverseIFEval'` 的分组配置，以 `name` 创建一个新的分组 abbr `InverseIFEval`。

3. 该分组配置了 `weights` 且没有显式配置 `metric`，因此加权聚合结果使用默认指标名 `weighted_average`。聚合完成后，内部结果相当于：

   ```python
   parsed_results[model_abbr]['InverseIFEval']['weighted_average'] = score
   ```

4. 生成表格时，`dataset_abbrs` 中的 `['InverseIFEval', 'weighted_average']` 先按第一个元素查找分组 abbr，再按第二个元素查找该分组下的指标，并将找到的分数写成一行。

因此，第一个元素既可以指向主配置中的原始 Dataset abbr，也可以指向 `summary_groups[*].name` 创建的分组 abbr；第二个元素则必须是该 abbr 实际拥有的指标名。

上例中的 `dataset_abbrs` 展开后，表格先显示三行聚合结果，再显示 16 个原始子集的 `accuracy`：

```text
dataset                    metric
InverseIFEval              weighted_average
InverseIFEval_zh           weighted_average
InverseIFEval_macro        naive_average
InverseIFEval_zh_QC        accuracy
InverseIFEval_zh_ITF       accuracy
...                        ...
InverseIFEval_en_CA        accuracy
```

如果完全省略 `dataset_abbrs`，`DefaultSummarizer` 会先按主配置中 `datasets` 的顺序输出每个 Dataset 的全部数值指标，再追加 `summary_groups` 生成且尚未出现的全部分组指标。因此，`dataset_abbrs` 的作用是筛选和排序展示结果，不是定义分组或计算公式，也不能重命名指标。

假设只有两个子集，`accuracy` 分别为 60 和 90，样本数分别为 100 和 200，那么：

```text
naive_average    = (60 + 90) / 2 = 75
weighted_average = (60 × 100 + 90 × 200) / (100 + 200) = 80
```

若 `subsets` 中任一 `[abbr, metric]` 找不到对应结果，整个分组会被标记为 `error: missing metrics`，不会使用剩余子集计算一个不完整的分数。

### 其他 summary group 字段

推荐像上例一样在 `subsets` 中使用 `[abbr, metric]`，明确指定每个输入指标。其他字段和写法的含义如下：

| 字段                            | 含义                                                                                                                                              |
| ------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------- |
| `name`                          | 新生成的分组 abbr。                                                                                                                               |
| `subsets`                       | 聚合输入。推荐使用 `[Dataset abbr, metric]` 二元组；也可以全部使用 Dataset abbr 字符串，但两种形式不能混用。                                      |
| `metric`                        | 聚合结果的指标名，不是输入指标名。输入指标仍由 `subsets` 的二元组指定。省略时根据聚合方式生成 `naive_average`、`weighted_average` 等名称。        |
| `weights`                       | `{Dataset abbr: 权重}`；配置后计算加权平均。使用二元组输入时，键也可以写成 `Dataset abbr@metric`。                                                |
| `std` / `sum` / `harmonic_mean` | 分别计算总体标准差、和或调和平均。为避免聚合方式互相影响，同一个分组只应选择其中一种，并且不要与 `weights` 同时使用。                             |
| `transforms`                    | `{Dataset abbr: '表达式'}`；在聚合前变换相应输入值，表达式中的 `x` 是原值。它可用于统一指标方向或量纲，但内部通过 `eval` 执行，只应使用可信配置。 |

例如 PluginEval 使用顶层 `metric` 给聚合结果命名，真正的两个输入指标仍写在 `subsets` 中：

```python
dict(
    name='plugin_eval-instruct_v1',
    metric='format_metric',
    subsets=[
        ['plugin_eval-instruct_v1', 'string_format_metric'],
        ['plugin_eval-instruct_v1', 'json_format_metric'],
    ],
)
```

该配置计算两个输入值的算术平均，并将结果保存为 `plugin_eval-instruct_v1/format_metric`。

当 `subsets` 全部是字符串时，`DefaultSummarizer` 会聚合所有子集共有的每个指标；此外还会生成一个默认聚合指标，该指标取每个子集按 [`METRIC_WHITELIST`](https://github.com/open-compass/opencompass/blob/main/opencompass/summarizers/default.py#L19) 排序后的首个指标作为输入。不同子集的首个指标可能不同，因此只有在各子集的主指标口径确定一致时才建议使用这种简写。

### 什么时候写自定义 Summarizer 类

只有在默认表格无法表达分组和聚合口径时才新增 Summarizer：继承 `DefaultSummarizer` 并覆盖 `summarize(output_path, time_str)`（仓库内的 [`CircularSummarizer`](https://github.com/open-compass/opencompass/blob/main/opencompass/summarizers/circular.py#L11)、[`MultiFacetedSummarizer`](https://github.com/open-compass/opencompass/blob/main/opencompass/summarizers/multi_faceted.py#L14) 等为例），配置中写 `type=MySummarizer`。综合分数必须说明：

- 宏平均还是按样本数加权；
- 缺失子集怎样处理；
- 指标方向和缩放是否一致；
- 重复运行如何聚合。

推荐先在 `opencompass/configs/summarizers/` 中用配置表达分组；确需新行为时再实现并注册 Summarizer 类。验收时用一个小型固定结果集验证汇总值，而不是依赖完整模型推理。
