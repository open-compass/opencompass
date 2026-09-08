# 理解输出与结果汇总

OpenCompass 默认把每次实验放在 `<work_dir>/<时间戳>/`。运行时实际生效的配置和每个阶段的产物应一起保留。

```text
<work_dir>/<时间戳>/
├── configs/       # 最终配置快照
├── logs/          # 推理与评测任务日志
├── predictions/   # 模型逐样本输出
├── results/       # Evaluator 产生的指标与明细
└── summary/       # Summarizer 产生的表格和汇总文件
```

## Predictions

预测文件位于 `predictions/<模型 abbr>/<数据集 abbr>.json`，以样本序号为键，是排查问题的第一证据。抽查时应同时看模型最终输入、原始回复、处理后的预测、样本索引以及错误字段。分数异常时，不要先调整 Summarizer；先确认样本是否加载正确、Prompt 是否完整、回复是否被截断。

```json
{
    "0": {
        "origin_prompt": [
            {"role": "user", "content": "一台印刷机 4 分钟印 36 页……它 10 分钟能印多少页？"}
        ],
        "prediction": "36 ÷ 4 = 9 页/分钟，10 × 9 = 90。#### 90",
        "gold": "90"
    },
    "1": {
        "origin_prompt": [
            {"role": "user", "content": "一件衣服原价 120 元打八折……"}
        ],
        "prediction": "120 × 0.8 = 96（元）",
        "gold": "96"
    }
}
```

`origin_prompt` 是最终送给模型的消息（对话式推理为 `role/content` 列表），`prediction` 是模型原始回复，`gold` 是参考答案。

## Results

结果文件位于 `results/<模型 abbr>/<数据集 abbr>.json`。Evaluator 读取预测与参考答案，执行答案抽取、规范化和指标计算。启用默认的 `--dump-eval-details` 时，结果还会包含逐样本明细；磁盘受限时可传 `--dump-eval-details False`。

```json
{
    "accuracy": 0.796875,
    "details": {
        "0": {"pred": "90", "answer": "90", "correct": true},
        "1": {"pred": "96", "answer": "96", "correct": true},
        "2": {"pred": "", "answer": "45", "correct": false}
    }
}
```

`details` 中每条记录的具体字段随 Evaluator 而异，上图为示意；对照明细可以定位是答案抽取失败还是模型答错。

只修改答案抽取或 Evaluator 时，通常可以通过 `--reuse <时间戳> --mode eval` 复用预测。

## Summary

Summarizer 控制结果的分组、顺序、别名及综合指标展示。默认汇总器可以直接输出基础表格：

```python
from opencompass.summarizers import DefaultSummarizer

summarizer = dict(type=DefaultSummarizer)
```

运行结束后，`summary/` 下会生成同一张表格的 `.txt`、`.csv` 和 `.md` 三个文件（如 `summary_20260908_141530.csv`），终端打印形如：

```text
dataset    version    metric    mode    qwen3.5-35b-a3b-vllm
---------  ---------  --------  ------  ---------------------
gsm8k      1d7fe4     accuracy  gen           79.69
```

`version` 来自数据集配置的版本标识，多模型评测时每个模型各占一列。

大型榜单一般继承 `opencompass/configs/summarizers/` 下的专用配置。综合分数可能采用宏平均、样本数加权或自定义分组。

只修改 Summarizer 时，可通过 `--reuse <时间戳> --mode viz` 重新汇总，无需重新推理和评分。
