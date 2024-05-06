# Using Large Models as JudgeLLM for Objective Evaluation

## Introduction

Traditional objective evaluations often rely on standard answers for reference. However, in practical applications, the predicted results of models may vary due to differences in the model's instruction-following capabilities or imperfections in post-processing functions. This can lead to incorrect extraction of answers and comparison with standard answers, resulting in potentially inaccurate evaluation outcomes. To address this issue, we have adopted a process similar to subjective evaluations by introducing JudgeLLM post-prediction to assess the consistency between model responses and standard answers. ([LLM-as-a-Judge](https://arxiv.org/abs/2306.05685)).

Currently, all models supported by the opencompass repository can be directly used as JudgeLLM. Additionally, we are planning to support dedicated JudgeLLMs.

## Currently Supported Objective Evaluation Datasets

1. MATH ([https://github.com/hendrycks/math](https://github.com/hendrycks/math))

## Custom JudgeLLM Objective Dataset Evaluation

OpenCompass currently supports most datasets that use `GenInferencer` for inference. The specific process for custom JudgeLLM objective evaluation includes:

1. Building evaluation configurations using API models or open-source models for inference of question answers.
2. Employing a selected evaluation model (JudgeLLM) to assess the outputs of the model.

### Step One: Building Evaluation Configurations, Using MATH as an Example

Below is the Config for evaluating the MATH dataset with JudgeLLM, with the evaluation model being *Llama3-8b-instruct* and the JudgeLLM being *Llama3-70b-instruct*. For more detailed config settings, please refer to `configs/eval_math_llm_judge.py`. The following is a brief version of the annotations to help users understand the meaning of the configuration file.

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


# ------------- Prompt Settings ----------------------------------------
# Evaluation template, please modify the template as needed, JudgeLLM typically uses [Yes] or [No] as the response. For the MATH dataset, the evaluation template is as follows:
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

# ------------- Inference Phase ----------------------------------------
# Models to be evaluated
models = [*hf_llama3_8b_instruct_model]
# Evaluation models
judge_models = hf_llama3_70b_instruct_model

eng_datasets = [*math_datasets]
chn_datasets = []
datasets = eng_datasets + chn_datasets


for d in eng_datasets:
    d['eval_cfg']= dict(
        evaluator=dict(
            type=LMEvaluator,
            # If you need to preprocess model predictions before judging,
            # you can specify a pred_postprocessor function here
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

# ------------- Evaluation Configuration --------------------------------
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

# Output folder
work_dir = 'outputs/obj_all/'
```

### Step Two: Launch Evaluation and Output Results

```shell
python run.py eval_math_llm_judge.py
```

This will initiate two rounds of evaluation. The first round involves model inference to obtain predicted answers to questions, and the second round involves JudgeLLM evaluating the consistency between the predicted answers and the standard answers, and scoring them.

- The results of model predictions will be saved in `output/.../timestamp/predictions/xxmodel/xxx.json`
- The JudgeLLM's evaluation responses will be saved in `output/.../timestamp/results/xxmodel/xxx.json`
- The evaluation report will be output to `output/.../timestamp/summary/timestamp/xxx.csv`

## Results

Using the Llama3-8b-instruct as the evaluation model and the Llama3-70b-instruct as the evaluator, the MATH dataset was assessed with the following results:

| Model               | JudgeLLM Evaluation | Naive Evaluation |
| ------------------- | ------------------- | ---------------- |
| llama-3-8b-instruct | 27.7                | 27.8             |
