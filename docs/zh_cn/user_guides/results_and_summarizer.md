# 理解输出与结果汇总

OpenCompass 默认将每次实验保存在 `<work_dir>/<时间戳>/` 目录下，其结构形如：

```text
outputs/my_eval/<时间戳>/
├── configs/       # 本次运行实际生效的配置快照
├── logs/
│   ├── infer/     # 推理任务日志
│   └── eval/      # 评测任务日志
├── predictions/   # Inferencer 产生的逐样本输出
├── results/       # Evaluator 产生的指标与明细
└── summary/       # Summarizer 产生的汇总文件
```

运行时实际生效的配置和各阶段产物应一起保留。后续使用 `--reuse` 时，OpenCompass 也是根据这些已有文件判断哪些任务可以跳过。

## 配置快照与日志

`configs/` 中保存的是 OpenCompass 解析配置并应用命令行覆盖项后生成的配置快照，而不只是原始配置文件的副本。排查结果差异时，应优先比较其中的模型、数据集、任务和 Summarizer 配置。

未启用 `--debug` 时，每个任务的标准输出和错误输出会写入 `logs/infer/` 或 `logs/eval/` 下的 `.out` 文件，文件路径同样按模型与数据集简称组织。启用 `--debug` 后，任务会在当前进程中执行，日志主要显示在终端，不一定生成对应的逐任务日志文件。

## Predictions：逐样本推理输出

预测文件位于 `predictions/<模型 abbr>/<数据集 abbr>.json`。文件名示例：

```text
predictions/gpt-6-astra-response/demo_gsm8k.json
```

文件以样本序号为键，是排查模型输入和原始输出的第一依据。典型结构如下：

```json
{
    "0": {
        "origin_prompt": [
            {"role": "user", "content": "一台印刷机 4 分钟印 36 页……它 10 分钟能印多少页？"}
        ],
        "prediction": "36 ÷ 4 = 9 页/分钟，10 × 9 = 90。#### 90",
        "gold": "计算过程……#### 90"
    },
    "1": {
        "origin_prompt": [
            {"role": "user", "content": "一件衣服原价 120 元，打八折后售价是多少？"}
        ],
        "prediction": "120 × 0.8 = 96，因此售价为 96 元。#### 96",
        "gold": "计算过程……#### 96"
    }
}
```

- `origin_prompt`：经过数据集模板和模型模板处理后送入模型的输入；具体结构取决于所用模板和模型类。
- `prediction`：Inferencer 保存的模型回复，尚未经过数据集 `eval_cfg.pred_postprocessor` 处理。
- `gold`：数据集 `reader_cfg.output_column` 中的原始参考答案。

部分情况下还会保存输入/输出长度、PPL、rollout 或多轮结果等字段，具体结构以实际预测文件为准。分数异常时，应先确认样本是否正确、Prompt 是否完整、回复是否被截断，再检查评分阶段。

## Results：评测指标与明细

结果文件位于 `results/<模型 abbr>/<数据集 abbr>.json`，文件名示例：

```text
results/gpt-6-astra-response/demo_gsm8k.json
```

评测阶段读取预测文件和原始数据，按照 `eval_cfg` 对预测与参考答案进行后处理，再调用 Evaluator 计算指标。默认启用 `--dump-eval-details`，以在结果中保存逐样本评测明细。

```json
{
    "accuracy": 79.6875,
    "details": [
        {"pred": "90", "answer": "90", "correct": true},
        {"pred": "", "answer": "45", "correct": false}
    ]
}
```

指标名称、数值范围以及 `details` 的结构都由 Evaluator 的具体实现决定。对照预测文件与明细，可以判断错误来自答案抽取、参考答案处理还是评分逻辑。

## Summary：结果汇总

Summarizer 读取 `results/` 中的指标，控制汇总表的展示顺序、指标选择和分组计算，但不会重新评分。配置示例：

```python
summarizer = dict(
    dataset_abbrs=[
        'All Results',
        ['demo_gsm8k', 'accuracy'],
    ],
)
```

当 `summarizer.type` 未设置时，入口会自动使用 `DefaultSummarizer`。显式写法与下面的配置等价：

```python
from opencompass.summarizers import DefaultSummarizer

summarizer = dict(
    type=DefaultSummarizer,
    dataset_abbrs=[
        'All Results',
        ['demo_gsm8k', 'accuracy'],
    ],
)
```

其中，`dataset_abbrs` 按列表顺序控制汇总表中的行：

- `['demo_gsm8k', 'accuracy']` 明确选择数据集简称和指标；
- 只写数据集简称时，显示该数据集优先级最高的数值指标；
- `All Results` 这类不对应数据集或分组的字符串会显示为分隔行；
- 不设置 `dataset_abbrs` 时，默认展示本次配置中所有可用的数据集指标和分组指标。

如果指定的数据集、指标或结果文件不存在，对应位置会显示 `-`。`dataset_abbrs` 只控制展示，不会让未参与本次实验的数据集产生结果。

如需对多个子集的结果进行聚合，可以配置 `summary_groups`。例如：

```python
summarizer = dict(
    dataset_abbrs=[
        ['demo_gsm8k', 'accuracy'],
        'reasoning-average',
    ],
    summary_groups=[
        dict(
            name='reasoning-average',
            subsets=[
                ['demo_gsm8k', 'accuracy'],
                ['another_dataset', 'accuracy'],
            ],
            transforms={
                # 假设该子集的 accuracy 取值范围为 0～1，聚合前先转换为百分制
                'another_dataset': 'x * 100',
            },
        ),
    ],
)
```

`name` 是汇总表中的分组名称，`subsets` 指定参与聚合的数据集与指标。`DefaultSummarizer` 支持以下聚合策略：

- 未指定策略时计算简单平均（宏平均），聚合指标名为 `naive_average`；
- `weights={'demo_gsm8k': 1, 'another_dataset': 2}` 按 `Σ(w·x) / Σw` 计算加权平均，聚合指标名为 `weighted_average`；
- `sum=True` 计算各子集分数之和；
- `std=True` 计算总体标准差；
- `harmonic_mean=True` 计算调和平均，只适用于大于 0 的分数。

`transforms` 用于在聚合前分别转换子集分数。键为子集 `abbr`，值为计算表达式，其中 `x` 表示该子集的原始分数。可用于百分制转换，或例如 `'max((3 - x) / 3, 0) * 100'` 的复杂表达式转化。转换只影响汇总计算，不会修改 `results/` 中的原始指标。

任一必需子集或指标缺失时，整个分组都会标记为缺失，不会仅对剩余结果进行计算。大批量评测时，可优先复用 `opencompass/configs/summarizers/` 下提供的预设配置。

运行结束后，`summary/` 下会生成同一张表格的 `.txt`、`.csv` 和 `.md` 文件（如 `summary_20260908_141530.csv`），并在终端打印类似结果：

```text
dataset       version    metric    mode    gpt-6-astra-response
------------  ---------  --------  ------  --------------------
demo_gsm8k    1d7fe4     accuracy  gen                    79.69
```

- `dataset`：数据集或分组的 `abbr`；
- `version`：根据数据集 `infer_cfg` 计算得到的提示词配置哈希的前 6 位，用于区分不同推理配置，并不是数据集发布版本；
- `metric`：Evaluator 输出的指标名称；
- `mode`：根据 Inferencer 类型标记为 `gen`、`ppl`、`ll` 或 `unknown`；
- 后续各列：模型 `abbr` 及其分数，每个模型占一列。
