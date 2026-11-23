# 答案抽取、后处理与评测器选择

评测方法由问题形式、模型能力和结果用途共同决定。OpenCompass 将“产生预测”和“给预测打分”拆开，同一批预测可以在不重新请求模型的情况下更换答案抽取或 Evaluator。

生成式评测通过 `GenInferencer` 获取回复，再用规则、数值等价、文本指标或执行结果评分；PPL 评测通过 `PPLInferencer` 比较候选答案概率，只适用于能够提供 token 概率的模型后端。开放式答案还可以使用 [LLM Judge](llm_judge.md) 或[级联评测](cascade_evaluator.md)。数学、代码、主观和多模态任务则需要相应的专项评测方法。

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

### 后处理器接口要求

`my_pred_postprocess` 和 `my_reference_postprocess` 都是逐样本调用的函数。以 `my_pred_postprocess` 为例，接口需要满足以下约束：

- 第一个位置参数接收单条模型输出，通常是 `str`；若预测结果是多候选列表，OpenCompass 会对列表中的每条字符串分别调用该函数。
- `dict(type=..., key=value)` 中除 `type` 外的字段会作为关键字参数传入函数，因此函数签名可以按需声明额外参数，例如 `def my_pred_postprocess(text: str, option: str = 'ABCD')`。
- 返回值应是一条规范化后的预测，而不是整批预测；返回类型需要与 `evaluator` 以及处理后的参考答案兼容，常见返回值是选项字母、短字符串、数字或任务自定义结构。
- 抽取失败时应返回一个可被 Evaluator 明确处理的值，例如原文、空字符串或约定的无效值。若希望 `--dump-extract-rate` 将失败样本统计为抽取失败，应返回 `''`、`None` 等空值；返回原文通常会作为普通预测继续评分。只有希望中断评测时才应抛出异常。

`dataset_postprocessor` 的接口相同，但它的输入来自数据集 reader 的输出列，也就是参考答案。

并非每个 Dataset 都需要两个后处理器。若参考答案已经是规范标签，可以省略 `dataset_postprocessor`。

### 现有配置示例

现有配置中也有同时使用这四个字段的例子。例如 [opencompass/configs/datasets/bbh/bbh_new_gen.py](https://github.com/open-compass/opencompass/blob/main/opencompass/configs/datasets/bbh/bbh_new_gen.py#L53-L57) 中，`pred_postprocessor` 和 `dataset_postprocessor` 都使用 `bbh_mcq_postprocess`，再交给 `BBHEvaluator_mcq` 评分：

```python
bbh_eval_cfg = dict(
    evaluator=dict(type=BBHEvaluator_mcq),
    pred_role='BOT',
    pred_postprocessor=dict(type=bbh_mcq_postprocess),
    dataset_postprocessor=dict(type=bbh_mcq_postprocess))
```

`bbh_mcq_postprocess` 的实现见 [opencompass/datasets/bbh.py](https://github.com/open-compass/opencompass/blob/main/opencompass/datasets/bbh.py#L32-L44)，代码如下：

```python
@TEXT_POSTPROCESSORS.register_module('bbh-mcq')
def bbh_mcq_postprocess(text: str) -> str:
    ans = text
    ans_line = ans.split('answer is ')
    if len(ans_line) != 1:
        ans = ans_line[1].strip()
    match = re.search(r'\(([A-Z])\)*', ans)
    if match:
        return match.group(1)
    match = re.search(r'([A-Z])', ans)
    if match:
        return match.group(1)
    return ans
```

它先尝试截取 `answer is ` 之后的内容，再提取 `(A)` 这类括号中的大写选项；如果没有括号形式，则提取第一个大写字母；仍然匹配不到时返回原文本。

## 验证自定义后处理器

如果新增或修改 `pred_postprocessor` / `dataset_postprocessor`，建议至少覆盖正确格式、带解释文本、多个候选答案、空回复、截断回复、Unicode/全半角差异和恶意格式。后处理器越宽松，越可能误判；越严格，越可能把语义正确答案判错。规则和测试样本应一起审阅。

## 为数据集选择 Evaluator

在 OpenCompass 中，评测方式通常由数据集配置中的 `eval_cfg.evaluator` 决定。对于 OpenCompass 已支持的数据集，建议直接使用对应 config 文件里配置好的 Evaluator 类和后处理器；这些配置通常已经对齐了数据集的答案格式、抽取规则和统计口径。

只有新增数据集、复用预测调整评分口径，或确认现有配置不符合任务需求时，才需要手动选择或实现 Evaluator。此时可以先从任务输出形态判断后处理器和 Evaluator 应该接收什么格式的数据：

- 选择题和分类通常比较规范化后的选项或类别标签，常见统计口径是 Accuracy；
- 短答案通常比较规范化后的短文本，可能使用 Exact Match、F1 或任务专用等价判断；
- 翻译和摘要通常保留较完整的生成文本，再使用 BLEU、ROUGE 等文本生成指标；
- 数学任务通常需要先抽取最终答案，再做数值、表达式或符号等价判断；
- 代码任务通常需要先抽取可执行代码，再通过隔离执行统计通过率或 pass@k；
- 开放式回答通常需要规则评分、专用模型或 LLM Judge。

同名指标也可能有大小写、空格、标点、多参考答案和平均方式差异。如果需要改动默认配置，应同时说明具体 Dataset 配置、Evaluator 类、后处理器和关键参数，而不只写指标名称。

## 抽取失败

答案抽取失败通常应作为错误计入，而不是丢弃样本。运行时可增加 `--dump-extract-rate`，并通过默认开启的逐样本评测明细检查失败案例：

```bash
opencompass my_eval.py --dump-extract-rate
```

`--dump-extract-rate` 的统计逻辑见 [opencompass/tasks/openicl_eval.py](https://github.com/open-compass/opencompass/blob/main/opencompass/tasks/openicl_eval.py#L437-L454)。它读取逐样本 details 中的 `predictions` 字段；如果该字段为空值，例如 `''`、`None` 或空列表，就计为抽取失败。对于生成式评测，details 中的 `predictions` 通常来自 Evaluator 返回的 `details[i]['pred']`，见 [opencompass/tasks/openicl_eval.py](https://github.com/open-compass/opencompass/blob/main/opencompass/tasks/openicl_eval.py#L499-L505)。

因此，抽取失败如何被统计取决于后处理器和 Evaluator 的约定。如果后处理器失败时返回 `''` 或 `None`，它会进入 `extract_rate` 的失败统计；如果像上面的 `bbh_mcq_postprocess` 一样返回原文本，只要原文本非空，就不会被 `extract_rate` 视为抽取失败，而是作为普通预测交给 Evaluator 判分。

如果修改了抽取规则，可复用已有预测只重跑评分：

```bash
opencompass my_eval.py --work-dir outputs/my_eval \
    --reuse <时间戳> --mode eval
```
