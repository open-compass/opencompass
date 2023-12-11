from mmengine.config import read_base
with read_base():
    from .models.qwen.hf_qwen_7b_chat import models as hf_qwen_7b_chat
    from .models.qwen.hf_qwen_14b_chat import models as hf_qwen_14b_chat
    from .models.chatglm.hf_chatglm3_6b import models as hf_chatglm3_6b
    from .models.baichuan.hf_baichuan2_7b_chat import models as hf_baichuan2_7b
    from .models.hf_internlm.hf_internlm_chat_7b import models as hf_internlm_chat_7b
    from .models.hf_internlm.hf_internlm_chat_20b import models as hf_internlm_chat_20b
    from .datasets.subjective_cmp.subjective_corev2 import subjective_datasets

datasets = [*subjective_datasets]

from opencompass.models import HuggingFaceCausalLM, HuggingFace, OpenAI
from opencompass.partitioners import NaivePartitioner
from opencompass.partitioners.sub_naive import SubjectiveNaivePartitioner
from opencompass.runners import LocalRunner
from opencompass.runners import SlurmSequentialRunner
from opencompass.tasks import OpenICLInferTask
from opencompass.tasks.subjective_eval import SubjectiveEvalTask
from opencompass.summarizers import Corev2Summarizer
models = [*hf_baichuan2_7b, *hf_chatglm3_6b, *hf_internlm_chat_20b, *hf_qwen_7b_chat, *hf_internlm_chat_7b, *hf_qwen_14b_chat]

api_meta_template = dict(
    round=[
        dict(role='HUMAN', api_role='HUMAN'),
        dict(role='BOT', api_role='BOT', generate=True)
    ],
    reserved_roles=[
        dict(role='SYSTEM', api_role='SYSTEM'),
    ],
)

infer = dict(
    partitioner=dict(type=NaivePartitioner),
    runner=dict(
        type=SlurmSequentialRunner,
        partition='llmeval',
        quotatype='auto',
        max_num_workers=256,
        task=dict(type=OpenICLInferTask)),
)


_meta_template = dict(
    round=[
        dict(role="HUMAN", begin='\n<|im_start|>user\n', end='<|im_end|>'),
        dict(role="BOT", begin="\n<|im_start|>assistant\n", end='<|im_end|>', generate=True),
    ],
)


judge_model =    dict(
        type=HuggingFaceCausalLM,
        abbr='qwen-7b-chat-hf',
        path="Qwen/Qwen-7B-Chat",
        tokenizer_path='Qwen/Qwen-7B-Chat',
        model_kwargs=dict(
            device_map='auto',
            trust_remote_code=True
        ),
        tokenizer_kwargs=dict(
            padding_side='left',
            truncation_side='left',
            trust_remote_code=True,
            use_fast=False,),
        pad_token_id=151643,
        max_out_len=2048,
        max_seq_len=2048,
        batch_size=8,
        meta_template=_meta_template,
        run_cfg=dict(num_gpus=1, num_procs=1),
    )


eval = dict(
    partitioner=dict(
        type=SubjectiveNaivePartitioner,
        mode='m2n',
        base_models = [*hf_baichuan2_7b, *hf_chatglm3_6b],
        compare_models = [*hf_baichuan2_7b, *hf_qwen_7b_chat, *hf_chatglm3_6b, *hf_qwen_14b_chat]
    ),
    runner=dict(
        type=SlurmSequentialRunner,
        partition='llmeval',
        quotatype='auto',
        max_num_workers=256,
        task=dict(
            type=SubjectiveEvalTask,
        judge_cfg=judge_model
        )),
)
work_dir = './corev2/'

summarizer = dict(
    type=Corev2Summarizer,
    match_method='smart',
)