# 级联评测

## 简介

规则评测（正则抽取、精确匹配、数学等价）成本低、结果稳定、可复现，但遇到答案形式多变的数据集时召回不足；LLM 评判灵活，却要为每条样本消耗一次模型请求。`CascadeEvaluator` 把两者组合成一个评测器：**先用规则评测器给所有样本打分，再把规则判错的样本交给 LLM 评判复核**，最终分数由两层结果合并得到。

典型场景是数学类数据集：`MATHVerifyEvaluator` 无法从预测中抽出 `\boxed{}`、或符号等价判断失败时，由 judge 按题意复核，避免"答案正确但形式不匹配"被误判。

级联评测器有两种工作模式，由 `parallel` 参数控制：

- **级联模式**（`parallel=False`）：只把规则判错的样本送 LLM。最终判对 = 规则判对 ∪（规则判错 & LLM 判对），分数只会相对纯规则口径向上修正，且 judge 请求量与规则错误数成正比，是最省成本的方式。
- **并行模式**（`parallel=True`，默认）：全部样本都送 LLM，规则或 LLM 任一判对即算对。相当于把规则当作 judge 之外的一条额外意见，宽容度最高，但成本与纯 LLM 评判相同。

## 运行机制

评测任务调用 `score()` 后，内部按以下流程执行：

1. **逐样本初始评测**：如果提供了 `rule_evaluator`，会先应用其 `pred_postprocess`，再调用规则评测器或 `sample_score_fn` 对单条样本打分；如果只提供 `sample_score_fn`，则直接使用原始预测打分。结果记入该样本的 `rule_evaluation`；日志会输出初始评测准确率（`Rule-based evaluation: ...`）。
2. **收集待复核样本**：级联模式收集初始评测判错的样本，并行模式收集全部样本；日志中的 `Samples requiring LLM evaluation (...)` 即待复核数量。
3. **构造评判子集**：从原测试集中 `select` 出待复核样本，并附加 `prediction`、`reference` 两列。`llm_evaluator` 的 `dataset_cfg` 会被自动置空，直接使用这个子集，因此 judge 模板可以引用题目、参考答案、模型预测等全部数据列。
4. **LLM 评判**：调用 `llm_evaluator.score()`——内部就是对 judge 模型的一次完整 `GenInferencer` 推理，结果写入 `<结果目录>_llm_judge_replica<N>.json`（N 为重复运行编号）。
5. **判定与汇总**：从 judge 明细中提取判定，按模式合并出最终 `accuracy`，并为每条样本写入 `cascade_correct`。

第 5 步的判定口径依次检查 judge 明细中的 `prediction` / `llm_judge` 字段是否为 `"A"` 或以 `"CORRECT"` 开头，其次是 `correct` 布尔值和 `score > 0.5`。因此 **judge 模板必须要求模型只回复 A / B**（或 CORRECT / INCORRECT），与 [LLM 作为评判器](llm_judge.md)中的模板约定一致；judge 输出成段解释文字时会被判为错误。

## 配置说明

`CascadeEvaluator` 的参数：

| 参数              | 类型                                      | 说明                                                    |
| ----------------- | ----------------------------------------- | ------------------------------------------------------- |
| `llm_evaluator`   | dict，必填                                | LLM 评判器配置，通常是 `GenericLLMEvaluator`            |
| `rule_evaluator`  | dict                                      | 规则评测器配置，如 `MATHVerifyEvaluator`                |
| `sample_score_fn` | `Callable[[str, str, Any], dict \| bool]` | 自定义单样本打分函数，返回含 `correct` 的 dict 或布尔值 |
| `parallel`        | bool，默认 `True`                         | `False` 为级联模式，`True` 为并行模式                   |

`rule_evaluator` 与 `sample_score_fn` 至少提供一个（否则初始化即报错）。若提供 `rule_evaluator`，评分前会先应用其 `pred_postprocess`；若只提供 `sample_score_fn`，则直接使用原始预测调用 `sample_score_fn`。如果同时提供二者，`sample_score_fn` 优先负责单样本打分，且收到的是经过 `rule_evaluator.pred_postprocess` 处理后的预测。

`sample_score_fn` 的调用形式是 `sample_score_fn(prediction, reference, test_item)`：`prediction` 是单条模型预测，`reference` 是对应参考答案，`test_item` 是原始测试样本（没有传入 `test_set` 时为 `None`）。如果返回 dict，至少应包含 `correct` 字段，例如 `{'correct': True, 'pred': prediction, 'answer': reference}`；其他字段会保留在逐样本评测明细中。如果返回非 dict，框架会用 `bool(result)` 转成 `correct`，并自动补充 `pred` 和 `answer`。

LLM 评判所需的测试集由评测任务自动传入，`llm_evaluator.dataset_cfg` 只是为了满足 `GenericLLMEvaluator` 的构造要求，级联评测时会被置空、不会重复加载数据集。judge 模型通过 `judge_cfg` 指定：留空（`dict()`）时读取环境变量 `OC_JUDGE_MODEL` / `OC_JUDGE_API_KEY` / `OC_JUDGE_API_BASE`，也可以填入任意的模型配置（本地或接口模型均可）。

仓库中的 [AIME 2026 级联评测配置](https://github.com/open-compass/opencompass/blob/main/opencompass/configs/datasets/aime2026/aime2026_cascade_eval_rawprompt_gen_0970dd.py) 是一个完整示例：规则层使用 `MATHVerifyEvaluator`，规则判错的样本再由 `GenericLLMEvaluator` 复核，judge 模型通过环境变量配置。

```python
from opencompass.openicl.icl_raw_prompt_template import RawPromptTemplate
from opencompass.openicl.icl_retriever import ZeroRetriever
from opencompass.openicl.icl_inferencer import GenInferencer
from opencompass.datasets import CustomDataset
from opencompass.datasets import generic_llmjudge_postprocess
from opencompass.evaluator import (
    CascadeEvaluator,
    GenericLLMEvaluator,
    MATHVerifyEvaluator
)

aime2026_reader_cfg = dict(input_columns=['problem'], output_column='answer')

aime2026_infer_cfg = dict(
    prompt_template=dict(
        type=RawPromptTemplate,
        messages=[
            {'role': 'user', 'content': '{problem}\nRemember to put your final answer within \\boxed{}.'},
        ],
    ),
    retriever=dict(type=ZeroRetriever),
    inferencer=dict(type=GenInferencer),
)

GRADER_TEMPLATE = """
    Please as a grading expert, judge whether the final answers given by the candidates below are consistent with the standard answers, that is, whether the candidates answered correctly.

    Here are some evaluation criteria:
    1. Please refer to the given standard answer. You don't need to re-generate the answer to the question because the standard answer has been given. You only need to judge whether the candidate's answer is consistent with the standard answer according to the form of the question. Don't try to answer the original question. You can assume that the standard answer is definitely correct.
    2. Because the candidate's answer may be different from the standard answer in the form of expression, before making a judgment, please understand the question and the standard answer first, and then judge whether the candidate's answer is correct, but be careful not to try to answer the original question.
    3. Some answers may contain multiple items, such as multiple-choice questions, multiple-select questions, fill-in-the-blank questions, etc. As long as the answer is the same as the standard answer, it is enough. For multiple-select questions and multiple-blank fill-in-the-blank questions, the candidate needs to answer all the corresponding options or blanks correctly to be considered correct.
    4. Some answers may be expressed in different ways, such as some answers may be a mathematical expression, some answers may be a textual description, as long as the meaning expressed is the same. And some formulas are expressed in different ways, but they are equivalent and correct.
    5. If the prediction is given with \\boxed{}, please ignore the \\boxed{} and only judge whether the candidate's answer is consistent with the standard answer.

    Please judge whether the following answers are consistent with the standard answer based on the above criteria. Grade the predicted answer of this new question as one of:
    A: CORRECT
    B: INCORRECT
    Just return the letters "A" or "B", with no text around it.

    Here is your task. Simply reply with either CORRECT, INCORRECT. Don't apologize or correct yourself if there was a mistake; we are just trying to grade the answer.


    <Original Question Begin>: \n{problem}\n<Original Question End>\n\n
    <Gold Target Begin>: \n{answer}\n<Gold Target End>\n\n
    <Predicted Answer Begin>: \n{prediction}\n<Predicted End>\n\n

    Judging the correctness of candidates' answers:
""".strip()

cascade_evaluator = dict(
    type=CascadeEvaluator,
    rule_evaluator=dict(
        type=MATHVerifyEvaluator,
    ),
    llm_evaluator=dict(
        type=GenericLLMEvaluator,
        prompt_template=dict(
            type=RawPromptTemplate,
            messages=[
                {'role': 'system', 'content': "You are a helpful assistant who evaluates the correctness and quality of models' outputs."},
                {'role': 'user', 'content': GRADER_TEMPLATE},
            ],
        ),
        dataset_cfg=dict(
            type=CustomDataset,
            path='opencompass/aime2026',
            reader_cfg=aime2026_reader_cfg,
        ),
        judge_cfg=dict(),
        dict_postprocessor=dict(type=generic_llmjudge_postprocess),
    ),
    parallel=False,
)
aime2026_eval_cfg = dict(
    evaluator=cascade_evaluator,
)

aime2026_datasets = [
    dict(
        type=CustomDataset,
        abbr='aime2026',
        path='opencompass/aime2026',
        reader_cfg=aime2026_reader_cfg,
        infer_cfg=aime2026_infer_cfg,
        eval_cfg=aime2026_eval_cfg,
        n=1,
    )
]
```

其中，`aime2026_reader_cfg` 指定题目列 `problem` 和参考答案列 `answer`；`GRADER_TEMPLATE` 将题目、参考答案与预测分别填入 `{problem}`、`{answer}` 和 `{prediction}`，并要求 judge 输出可被判定逻辑识别的 `A` / `B`（同时也接受 `CORRECT` / `INCORRECT`）。`cascade_evaluator` 中的 `rule_evaluator` 是第一层数学等价评测，`llm_evaluator` 是第二层复核；`parallel=False` 表示只复核第一层判错的样本。`dict_postprocessor` 使用 `generic_llmjudge_postprocess` 将 judge 输出整理为评测明细。空的 `judge_cfg` 表示从 `OC_JUDGE_*` 环境变量读取 judge 模型配置。

## 评估输出

结果文件中除 `accuracy` 外还包含 `cascade_stats` 与逐样本 `details`：

```python
{
    'accuracy': 85.0,
    'cascade_stats': {
        'total_samples': 100,
        'rule_correct': 70,
        'rule_accuracy': 70.0,
        'llm_evaluated': 30,
        'llm_correct': 15,
        'llm_accuracy': 50.0,
        'final_correct': 85,
        'final_accuracy': 85.0,
        'parallel_mode': False,
    },
    'details': [
        # ... 每个样本一项
    ],
}
```

各字段含义：

- `rule_accuracy`：纯规则口径的准确率，也是级联模式下最终分数的下限；
- `llm_evaluated`：实际送 judge 的样本数（级联模式 = 规则判错数，并行模式 = 全体）；
- `llm_accuracy`：judge 在这些样本上的判对率，可用于估计规则评测器的漏判规模；
- `final_accuracy`：两层合并后的最终上报指标。

每条样本的 `details` 结构：

```python
{
    'rule_evaluation': {'correct': False, ...},                # 规则层明细
    'llm_evaluation': {'prediction': 'A', 'llm_correct': True, ...},  # judge 明细（若有）
    'cascade_correct': True,                                   # 两层合并后的最终判定
}
```

启动任务时加 `--dump-eval-details` 可以把这些明细落盘，便于逐条复核规则与 judge 的分歧样本。

## 结果缓存与复跑

judge 结果保存在 `<结果目录>_llm_judge_replica<N>.json`。再次评测时若该文件已存在则直接加载、不再请求 judge；加载到的样本数与本次待复核样本数不一致时会报错并提示删除缓存——更换被测模型、修改规则评测器或答案后处理都会改变"规则判错样本集"，导致缓存失配。

重复运行（dataset replica）使用各自编号的缓存文件，互不干扰。

## 何时选择级联评测

- 已有规则评测器但召回不足（数学等价判断、开放式问答）→ **级联模式**，成本只花在规则失败的样本上；
- 希望规则和 judge 互相兜底、取判对的并集 → **并行模式**；
- 选择题等规则已完全可靠的场景 → 直接使用规则评测器，不需要 judge；
- 没有可用的 judge 服务或需要严格控制成本 → 纯规则评测。

与纯 LLM 评判相比，级联评测的分数可以拆解为"规则分 + judge 修正量"（见 `cascade_stats`）。对外报告时应同时给出两层口径，便于与他人用纯规则或纯 judge 得到的分数比较。

## 完整示例

完整配置见 [opencompass/configs/datasets/aime2026/aime2026_cascade_eval_rawprompt_gen_0970dd.py](https://github.com/open-compass/opencompass/blob/main/opencompass/configs/datasets/aime2026/aime2026_cascade_eval_rawprompt_gen_0970dd.py)。
