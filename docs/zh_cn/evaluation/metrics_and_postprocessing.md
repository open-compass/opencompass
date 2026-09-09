# 指标、答案抽取与后处理

模型原始回复通常不能直接与参考答案比较。客观评测一般经过三个步骤：从回复抽取答案、对预测和参考答案做规范化、由 Evaluator 聚合指标。

```text
原始回复 → pred_postprocessor → 规范预测
参考答案 → dataset_postprocessor → 规范答案
规范预测 + 规范答案 → Evaluator → 指标
```

## 在 Dataset 中配置

```python
eval_cfg = dict(
    pred_role='BOT',
    pred_postprocessor=dict(type=my_pred_postprocess),
    dataset_postprocessor=dict(type=my_reference_postprocess),
    evaluator=dict(type=MyEvaluator),
)
```

并非每个 Dataset 都需要两个后处理器。若参考答案已经是规范标签，可以省略 `dataset_postprocessor`。

## 选择指标

- 选择题和分类：Accuracy；
- 短答案：Exact Match、F1 或任务专用等价判断；
- 翻译：BLEU 等文本生成指标；
- 摘要：ROUGE 等重叠指标；
- 数学：数值、表达式或符号等价；
- 代码：隔离执行后的通过率与 pass@k；
- 开放式回答：规则评分、专用模型或 LLM Judge。

同名指标也可能有大小写、空格、标点、多参考答案和平均方式差异。发布结果时应记录具体 Evaluator 类和后处理器，而不只写指标名称。

## 抽取失败

答案抽取失败通常应作为错误计入，而不是丢弃样本。运行时可增加 `--dump-extract-rate`，并通过默认开启的逐样本评测明细检查失败案例：

```bash
opencompass my_eval.py --dump-extract-rate
```

如果修改了抽取规则，可复用已有预测只重跑评分：

```bash
opencompass my_eval.py --work-dir outputs/my_eval \
    --reuse <时间戳> --mode eval
```

## 验证新规则

至少覆盖正确格式、带解释文本、多个候选答案、空回复、截断回复、Unicode/全半角差异和恶意格式。后处理器越宽松，越可能误判；越严格，越可能把语义正确答案判错。规则和测试样本应一起审阅。
