from mmengine.config import read_base
with read_base():
    from .models.qwen.hf_qwen_7b_chat import models as hf_qwen_7b_chat
    from .models.chatglm.hf_chatglm3_6b import models as hf_chatglm3_6b
    from .models.qwen.hf_qwen_72b_chat import models as qwen72
    from .models.yi.hf_yi_34b_chat import models as yi34
    from .models.mixtral.hf_mixtral_8x7b_instruct_v0_1 import models as m7b8
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
from opencompass.summarizers import AllObjSummarizer
from opencompass.openicl.icl_evaluator import LMEvaluator
from opencompass.openicl.icl_prompt_template import PromptTemplate
# -------------Inferen Stage ----------------------------------------

chn_obj_prompt = """
请根据 问题，参考答案 和 模型回答 来判断模型是否回答正确

[问题开始]
{question}
[问题结束]

[参考答案开始]
{obj_gold}
[参考答案结束]

[模型回答开始]
{prediction}
[模型回答结束]

请判断模型是否回答正确，你需要按照如下格式输出你的评判结果及作出判断的理由：
如果你认为模型回答正确，则输出
"结果: [[正确]]
 理由: xxx
"
如果你认为模型回答错误，则输出
"结果: [[错误]]
 理由: xxx
"
"""

eng_obj_prompt = """
Please assess whether the model has answered correctly based on the question, reference answer, and model's response.

[Question Start]
{question}
[Question End]

[Reference Answer Start]
{obj_gold}
[Reference Answer End]

[Model's Response Start]
{prediction}
[Model's Response End]

Please determine whether the model has answered correctly. Output your judgment result and provide the reasoning in the following format:
If you believe the model's response is correct, output:
"Result: [[Correct]]
 Reason: xxx
"
If you believe the model's response is incorrect, output:
"Result: [[Incorrect]]
 Reason: xxx
"
"""

models = [*hf_chatglm3_6b, *hf_qwen_7b_chat]
eng_datasets = [*gsm8k_datasets]
chn_datasets = []
datasets = eng_datasets + chn_datasets
judge_model = qwen72[0]
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
        max_task_size=200000,
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
    type=AllObjSummarizer
)