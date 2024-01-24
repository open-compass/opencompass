from mmengine.config import read_base
with read_base():
    from .models.qwen.hf_qwen_7b_chat import models as hf_qwen_7b_chat
    from .models.chatglm.hf_chatglm3_6b import models as hf_chatglm3_6b
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

obj_prompt = """
请根据 问题，参考答案 和 模型回答 来判断模型是否回答正确

[问题]
{question}

[参考答案]
{obj_gold}

模型回答
{prediction}

如果模型回答正确，则输出[[1]]
如果模型回答错误，则输出[[0]]
"""

models = [*hf_chatglm3_6b, *hf_qwen_7b_chat]
all_datasets = [*gsm8k_datasets]
for d in all_datasets:
    d['eval_cfg']= dict(
        evaluator=dict(
            type=LMEvaluator,
            prompt_template=dict(
                type=PromptTemplate,
                template=dict(round=[
                    dict(
                        role='HUMAN',
                        prompt = obj_prompt
                    ),
                ]),
            ),
        ),
        pred_role="BOT",
    )
datasets = all_datasets

judge_model = models[1]

infer = dict(
    #partitioner=dict(type=NaivePartitioner),
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
    type=AlignmentBenchSummarizer, judge_type = 'general'
)

work_dir = 'outputs/obj_all/'