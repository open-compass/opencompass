# 结果展示

在评测完成后，评测的结果需要被打印到屏幕或者被保存下来，该过程是由 summarizer 控制的。

```{note}
如果 summarizer 出现在了 config 中，则评测结果输出会按照下述逻辑进行。
如果 summarizer 没有出现在 config 中，则评测结果会按照 `dataset` 中出现的顺序进行输出。
```

## 样例

一个典型的 summerizer 配置文件如下：

```python
summarizer = dict(
    dataset_abbrs = [
        'race',
        'race-high',
        'race-middle',
    ],
    summary_groups=[
        {'name': 'race', 'subsets': ['race-high', 'race-middle']},
    ]
)
```

其输出结果如下：

```text
dataset      version    metric         mode      internlm-7b-hf
-----------  ---------  -------------  ------  ----------------
race         -          naive_average  ppl                76.23
race-high    0c332f     accuracy       ppl                74.53
race-middle  0c332f     accuracy       ppl                77.92
```

summarizer 会以 config 中的 `models`, `datasets` 为全集，去尝试读取 `{work_dir}/results/` 路径下的评测分数，并按照 `summarizer.dataset_abbrs` 列表的顺序进行展示。另外，summarizer 会尝试通过 `summarizer.summary_groups` 来进行一些汇总指标的计算。当且仅当 `subsets` 中的值都存在时，对应的 `name` 指标才会生成，这也就是说，若有部分数字缺失，则这个汇总指标也是会缺失的。若分数无法通过上述两种方式被获取到，则 summarizer 会在表格中对应项处使用 `-` 进行表示。

此外，输出结果是有多列的：

- `dataset` 列与 `summarizer.dataset_abbrs` 配置一一对应
- `version` 列是这个数据集的 hash 值，该 hash 值会考虑该数据集模板的评测方式、提示词、输出长度限制等信息。用户可通过该列信息确认两份评测结果是否可比
- `metric` 列是指这个指标的评测方式，具体说明见 [metrics](./metrics.md)
- `mode` 列是指这个推理结果的获取方式，可能的值有 `ppl` / `gen`。对于 `summarizer.summary_groups` 的项，若被 `subsets` 的获取方式都一致，则其值也跟 `subsets` 一致，否则即为 `mixed`
- 其后若干列，一列代表一个模型

## 完整字段说明

summarizer 字段说明如下

- `dataset_abbrs`: (list，可选) 展示列表项。若该项省略，则会输出全部评测结果。
- `summary_groups`: (list，可选) 汇总指标配置。

`summary_groups` 中的字段说明如下：

- `name`: (str) 汇总指标的名称。
- `subsets`: (list) 被汇总指标的名称。注意它不止可以是原始的 `dataset_abbr`，也可以是另一个汇总指标的名称。
- `weights`: (list，可选) 被汇总指标的权重。若该项省略，则默认使用不加权的求平均方法。

注意，我们在 `configs/summarizers/groups` 路径下存放了 MMLU, C-Eval 等数据集的评测结果汇总，建议优先考虑使用。
