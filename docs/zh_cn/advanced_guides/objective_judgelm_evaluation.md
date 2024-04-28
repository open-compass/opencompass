# 用大模型做为JudgeLLM进行客观评测

## 介绍

通常的客观评测虽有标准答案作为参考，但是在实际应用中，模型预测结果可能因为模型指令遵循能力不同或后处理函数的不完善而产生差异，导致无法抽取到正确的答案并与标准答案进行对比。因此客观评测的结果可能并不完全准确。为了解决这一问题，我们参照主观评测，在预测完成后引入了JudgeLLM作为评价模型，以评估模型回答和标准答案的一致性。（[LLM-as-a-Judge](https://arxiv.org/abs/2306.05685)）。

目前opencompass仓库里支持的所有模型都可以直接作为JudgeLLM进行调用，此外一些专用的JudgeLLM我们也在计划支持中。

## 目前已支持的用JudgeLLM进行直接评测的客观评测数据集

1. MATH（https://github.com/hendrycks/math）

## 自定义JudgeLLM客观数据集评测

目前的OpenCompass支持大部分采用`GenInferencer`的数据集进行推理。自定义JudgeLLM客观评测的具体流程包括:

1. 构建评测配置，使用API模型或者开源模型进行问题答案的推理
2. 使用选定的评价模型(JudgeLLM)对模型输出进行评估

### 第一步：构建评测配置，以MATH为例

下面是对MATH数据集进行JudgeLLM评测的Config，评测模型为*Llama3-8b-instruct*，JudgeLLM为*Llama3-70b-instruct*。更详细的config setting请参考 `configs/eval_math_llm_judge.py`，下面我们提供了部分简略版的注释，方便用户理解配置文件的含义。

```python
# Most of the code in this file is copied from https://github.com/openai/simple-evals/blob/main/math_eval.py
from mmengine.config import read_base
with read_base():
    from .models.hf_llama.hf_llama3_8b_instruct import models as hf_llama3_8b_instruct_model # noqa: F401, F403
    from .models.hf_llama.hf_llama3_70b_instruct import models as hf_llama3_70b_instruct_model  # noqa: F401, F403
    from .datasets.math.math_llm_judge import math_datasets  # noqa: F401, F403
from opencompass.datasets import math_judement_preprocess
from opencompass.partitioners import NaivePartitioner, SizePartitioner
from opencompass.partitioners.sub_naive import SubjectiveNaivePartitioner
from opencompass.partitioners.sub_size import SubjectiveSizePartitioner
from opencompass.runners import LocalRunner
from opencompass.runners import SlurmSequentialRunner
from opencompass.tasks import OpenICLInferTask
from opencompass.tasks.subjective_eval import SubjectiveEvalTask
from opencompass.summarizers import AllObjSummarizer
from opencompass.openicl.icl_evaluator import LMEvaluator
from opencompass.openicl.icl_prompt_template import PromptTemplate


# ------------- Prompt设置 ----------------------------------------
# 评测模板，请根据需要修改模板，JudgeLLM默认采用[Yes]或[No]作为回答，在MATH数据集中，评测模板如下
eng_obj_prompt = """
Look at the following two expressions (answers to a math problem) and judge whether they are equivalent. Only perform trivial simplifications

Examples:

    Expression 1: $2x+3$
    Expression 2: $3+2x$

[Yes]

    Expression 1: 3/2
    Expression 2: 1.5

[Yes]

    Expression 1: $x^2+2x+1$
    Expression 2: $y^2+2y+1$

[No]

    Expression 1: $x^2+2x+1$
    Expression 2: $(x+1)^2$

[Yes]

    Expression 1: 3245/5
    Expression 2: 649

[No]
(these are actually equal, don't mark them equivalent if you need to do nontrivial simplifications)

    Expression 1: 2/(-3)
    Expression 2: -2/3

[Yes]
(trivial simplifications are allowed)

    Expression 1: 72 degrees
    Expression 2: 72

[Yes]
(give benefit of the doubt to units)

    Expression 1: 64
    Expression 2: 64 square feet

[Yes]
(give benefit of the doubt to units)

    Expression 1: 64
    Expression 2:

[No]
(only mark as equivalent if both expressions are nonempty)

---

YOUR TASK


Respond with only "[Yes]" or "[No]" (without quotes). Do not include a rationale.
    Expression 1: {obj_gold}
    Expression 2: {prediction}

"""

# -------------推理阶段 ----------------------------------------
# 需要评测的模型
models = [*hf_llama3_8b_instruct_model]
# 评价模型
judge_models = hf_llama3_70b_instruct_model

eng_datasets = [*math_datasets]
chn_datasets = []
datasets = eng_datasets + chn_datasets


for d in eng_datasets:
    d['eval_cfg']= dict(
        evaluator=dict(
            type=LMEvaluator,
            # 如果你需要在判断之前预处理模型预测，
            # 你可以在这里指定pred_postprocessor函数
            pred_postprocessor=dict(type=math_judement_preprocess),
            prompt_template=dict(
                type=PromptTemplate,
                template=dict(round=[
                    dict(
                        role='HUMAN',
                        prompt = eng_obj_prompt
                    ),
                ]),
            ),
        ),
        pred_role="BOT",
    )

infer = dict(
    partitioner=dict(type=SizePartitioner, max_task_size=40000),
    runner=dict(
        type=LocalRunner,
        max_num_workers=256,
        task=dict(type=OpenICLInferTask)),
)

# ------------- 评测配置 --------------------------------
eval = dict(
    partitioner=dict(
        type=SubjectiveSizePartitioner, max_task_size=80000, mode='singlescore', models=models, judge_models=judge_models,
    ),
    runner=dict(type=LocalRunner,
        max_num_workers=16, task=dict(type=SubjectiveEvalTask)),
)

summarizer = dict(
    type=AllObjSummarizer
)

# 输出文件夹
work_dir = 'outputs/obj_all/'
```

### 第二步 启动评测并输出评测结果

```shell
python run.py eval_math_llm_judge.py
```

此时会进行两轮评测，第一轮是模型推理得到问题的预测答案，第二轮是JudgeLLM评测预测答案和标准答案的一致性，并打分。

- 模型预测的结果会保存在 `output/.../timestamp/predictions/xxmodel/xxx.json`
- JudgeLLM的评测回复会保存在 `output/.../timestamp/results/xxmodel/xxx.json`
- 评测报告则会输出到 `output/.../timestamp/summary/timestamp/xxx.csv`

## 评测结果

采用Llama3-8b-instruct作为评价模型，Llama3-70b-instruct作为评价器，对MATH数据集进行评价，结果如下：

| Model               | JudgeLLM Evaluation | Naive Evaluation |
| ------------------- | ------------------- | ---------------- |
| llama-3-8b-instruct | 27.7                | 27.8             |
