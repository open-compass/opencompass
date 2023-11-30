from mmengine.config import read_base
with read_base():
    from .models.qwen.hf_qwen_7b_chat import models as hf_qwen_7b_chat
    from .models.chatglm.hf_chatglm2_6b import models as hf_chatglm2_6b
    from .models.hf_internlm.hf_internlm_chat_7b import models as hf_internlm_chat_7b
    from .datasets.subjective_cmp.subjective_cmp import subjective_datasets
    from .summarizers.subjective import summarizer

datasets = [*subjective_datasets]

from opencompass.models import HuggingFaceCausalLM, HuggingFace, OpenAI
from opencompass.partitioners.sub_naive import SubjectiveNaivePartitioner
from opencompass.runners import LocalRunner
from opencompass.tasks.subjective_eval import SubjectiveEvalTask

models = [*hf_qwen_7b_chat, *hf_chatglm2_6b, *hf_internlm_chat_7b]

api_meta_template = dict(
    round=[
        dict(role='HUMAN', api_role='HUMAN'),
        dict(role='BOT', api_role='BOT', generate=True)
    ],
    reserved_roles=[
        dict(role='SYSTEM', api_role='SYSTEM'),
    ],
)

eval = dict(
    partitioner=dict(
        type=SubjectiveNaivePartitioner,
        mode='all',  # 新参数
    ),
    runner=dict(
        type=LocalRunner,
        max_num_workers=2,  # 支持并行比较
        task=dict(
            type=SubjectiveEvalTask,  # 新 task，用来读入一对 model 的输入
            judge_cfg=dict(
                abbr='GPT4',
                type=OpenAI,
                path='gpt-4-0613',
                key='ENV',
                meta_template=api_meta_template,
                query_per_second=1,
                max_out_len=2048,
                max_seq_len=2048,
                batch_size=2),
        )),
)
