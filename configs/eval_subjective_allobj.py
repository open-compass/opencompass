from mmengine.config import read_base
with read_base():
    from .models.qwen.hf_qwen_7b_chat import models as hf_qwen_7b_chat
    from .models.chatglm.hf_chatglm3_6b import models as hf_chatglm3_6b
    from .models.qwen.hf_qwen_72b_chat import models as qwen72
    from .datasets.gsm8k.gsm8k_gen import gsm8k_datasets  # noqa: F401, F403
from opencompass.models import HuggingFaceCausalLM, HuggingFace, HuggingFaceChatGLM3
from opencompass.models.openai_api import OpenAIAllesAPIN
from opencompass.partitioners import NaivePartitioner, SizePartitioner
from opencompass.partitioners.sub_naive import SubjectiveNaivePartitioner
from opencompass.partitioners.sub_size import SubjectiveSizePartitioner
from opencompass.runners import LocalRunner
from opencompass.runners import SlurmSequentialRunner
from opencompass.tasks import OpenICLInferTask
from opencompass.tasks.subjective_eval import SubjectiveEvalTask
from opencompass.summarizers import AlignmentBenchSummarizer
from opencompass.openicl.icl_evaluator import LMEvaluator
from opencompass.openicl.icl_prompt_template import PromptTemplate
# -------------Inferen Stage ----------------------------------------

chn_obj_prompt = """
请根据 问题，参考答案 和 模型回答 来判断模型是否回答正确

[问题]
{question}

[参考答案]
{obj_gold}

[模型回答]
{prediction}

如果模型回答正确，则输出[[1]]
如果模型回答错误，则输出[[0]]
"""

eng_obj_prompt = """
Please determine whether the model has answered correctly based on the question, reference answer, and model response.

[Question]
{question}

[Reference Answer]
{obj_gold}

[Model Response]
{prediction}

Output [[1]] if the model answered correctly, and [[0]] if the model answered incorrectly.
"""

models = [*hf_chatglm3_6b, *hf_qwen_7b_chat]
eng_datasets = [*gsm8k_datasets]
chn_datasets = []
judge_model = qwen72[0]
judge_model = models[1]
work_dir = 'outputs/obj_all/'

for d in eng_datasets:
    d['eval_cfg']= dict(
        evaluator=dict(
            type=LMEvaluator,
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
for d in chn_datasets:
    d['eval_cfg']= dict(
        evaluator=dict(
            type=LMEvaluator,
            prompt_template=dict(
                type=PromptTemplate,
                template=dict(round=[
                    dict(
                        role='HUMAN',
                        prompt = chn_obj_prompt
                    ),
                ]),
            ),
        ),
        pred_role="BOT",
    )
datasets = eng_datasets + chn_datasets

infer = dict(
    partitioner=dict(type=SizePartitioner, max_task_size=10000),
    runner=dict(
        type=SlurmSequentialRunner,
        partition='llm_dev2',
        quotatype='auto',
        max_num_workers=256,
        task=dict(type=OpenICLInferTask)),
)

## ------------- Evaluation Configuration
eval = dict(
    partitioner=dict(
        type=SubjectiveSizePartitioner,
        strategy='split',
        max_task_size=10000,
        mode='singlescore',
        models = models
    ),
    runner=dict(
        type=SlurmSequentialRunner,
        partition='llm_dev2',
        quotatype='auto',
        max_num_workers=256,
        task=dict(
            type=SubjectiveEvalTask,
            judge_cfg=judge_model
        )),
)

summarizer = dict(
    type=AlignmentBenchSummarizer, judge_type = 'autoj'
)