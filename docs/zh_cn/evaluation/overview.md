# 评测方法总览

评测方法由问题形式、模型能力和结果用途共同决定。OpenCompass 将“产生预测”和“给预测打分”拆开，同一批预测可以在不重新请求模型的情况下更换答案抽取或 Evaluator。

## 客观评测

生成式评测通过 `GenInferencer` 获取模型回复，再用规则、数值等价、文本指标或执行结果评分。PPL 评测通过 `PPLInferencer` 比较候选答案概率，只适用于能够提供 token 概率的本地模型后端。

常见指标包括 Accuracy、Exact Match、F1、BLEU、ROUGE 和 pass@k。指标名称相同也不保证口径相同；答案规范化、抽取失败处理和子集加权都会改变最终值。

## 大模型评判

开放式答案可以由 Judge 模型进行点式评分、成对比较或分项打分。Judge 的模型版本、Prompt、候选顺序、温度、失败重试和聚合策略都属于评测配置的一部分。详见[大模型评判](llm_judge.md)。规则与 LLM 评判还可以级联组合——规则先行、判错样本交由 Judge 复核，见[级联评测](cascade_evaluator.md)。

## 专项评测

- 数学题需要可靠地抽取最终答案并判断符号或数值等价；
- 代码题需要在隔离环境中执行不受信任代码，并明确 pass@k 采样方式；
- 主观评测需要控制位置偏差、Judge 偏好和成对比较顺序；
- 长上下文评测必须记录 tokenizer、实际输入 token 数和截断情况；
- 多模态评测还需固定媒体预处理及官方评测器版本。

## 配置位置

数据集的 `eval_cfg` 通常包含：

```python
eval_cfg = dict(
    evaluator=dict(type=...),
    pred_postprocessor=dict(type=...),
    dataset_postprocessor=dict(type=...),
)
```

后处理先把自由文本变成可比较答案，Evaluator 再计算逐样本和聚合指标。Summarizer 只组织已有结果，不负责重新判题。常用指标配置参阅[指标与后处理](metrics_and_postprocessing.md)。
