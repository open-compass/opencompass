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


# -------------Prompt Settings ----------------------------------------
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

# -------------Inferen Stage ----------------------------------------
# eval models
models = [*hf_llama3_8b_instruct_model]
# judge models
judge_models = hf_llama3_70b_instruct_model

eng_datasets = [*math_datasets]
chn_datasets = []
datasets = eng_datasets + chn_datasets
work_dir = 'outputs/obj_all/'

for d in eng_datasets:
    d['eval_cfg']= dict(
        evaluator=dict(
            type=LMEvaluator,
            # If you need to preprocess the prediction before judging,
            # you can specify the pred_postprocessor function here
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
        pred_role='BOT',
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
