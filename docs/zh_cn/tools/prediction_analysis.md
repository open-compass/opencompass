# 预测结果检查与错误样本分析

## Case Analyzer

```bash
python tools/case_analyzer.py my_eval.py -w outputs/my_eval/<时间戳>
```

它读取已有预测和评测结果，整理错误样本及带标注的完整样本。使用前确认 `-w` 指向实际时间戳目录，而不是只指向包含多个实验的上级目录。

## 合并分片预测

```bash
python tools/prediction_merger.py my_eval.py \
    -w outputs/my_eval \
    -r <时间戳>
```

Prediction Merger 用于合并 Partitioner 产生的分片。`-w` 是包含各时间戳目录的工作目录，`-r` 选择具体时间戳（默认为 `latest`）。合并前不要改变 Dataset 简称、样本范围或切分策略；合并后检查样本 ID 是否重复或缺失。

## 人工抽查顺序

1. 对照 `configs/` 中的最终配置；
2. 检查失败和重试日志；
3. 比较模型输入、原始回复和后处理答案；
4. 检查空回复、截断、解析失败和异常超长输出；
5. 再判断问题来自模型、Prompt、Loader 还是 Evaluator。
