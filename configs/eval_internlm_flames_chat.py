from mmengine.config import read_base

from opencompass.models import HuggingFaceCausalLM
from opencompass.partitioners import NaivePartitioner
from opencompass.partitioners.sub_naive import SubjectiveNaivePartitioner
from opencompass.runners import LocalRunner
from opencompass.tasks import OpenICLInferTask
from opencompass.tasks.subjective_eval import SubjectiveEvalTask
from opencompass.summarizers import FlamesSummarizer


# -------------Inferen Stage ----------------------------------------

with read_base():
    from .datasets.flames.flames_gen import flames_datasets
    from .models.hf_internlm.hf_internlm2_chat_7b import models
datasets = [*flames_datasets]


from opencompass.models import HuggingFaceCausalLM


_meta_template = dict(
    round=[
        dict(role='HUMAN', begin='<|im_start|>user\n', end='<|im_end|>\n'),
        dict(role='BOT', begin='<|im_start|>assistant\n', end='<|im_end|>\n', generate=True),
    ],
)

models = [
    dict(
        type=HuggingFaceCausalLM,
        abbr='internlm2-chat-7b-hf',
        path='internlm/internlm2-chat-7b',
        tokenizer_path='internlm/internlm2-chat-7b',
        model_kwargs=dict(
            trust_remote_code=True,
            device_map='auto',
        ),
        tokenizer_kwargs=dict(
            padding_side='left',
            truncation_side='left',
            use_fast=False,
            trust_remote_code=True,
        ),
        max_out_len=2048,
        max_seq_len=2048,
        batch_size=8,
        meta_template=_meta_template,
        run_cfg=dict(num_gpus=1, num_procs=1),
        end_str='<|im_end|>',
        generation_kwargs = {'eos_token_id': [2, 92542], 'do_sample': True},
        batch_padding=True,
    )
]


infer = dict(
    partitioner=dict(type=NaivePartitioner),
    runner=dict(
        type=LocalRunner,
        max_num_workers=256,
        task=dict(type=OpenICLInferTask)),
)


# -------------Evalation Stage ----------------------------------------


## ------------- JudgeLLM Configuration---------------------------------
internlm1_chat_template = dict(
    round=[
        dict(role='HUMAN', begin='<|User|>:', end='\n'),
        dict(role='BOT', begin='<|Bot|>:', end='<eoa>\n', generate=True),
    ],
)

judge_models = [
    dict(
        type=HuggingFaceCausalLM,
        abbr='flames-scorer',
        path='CaasiHUANG/flames-scorer',
        tokenizer_path='CaasiHUANG/flames-scorer',
        model_kwargs=dict(
            trust_remote_code=True,
            device_map='auto',
        ),
        tokenizer_kwargs=dict(
            padding_side='left',
            truncation_side='left',
            use_fast=False,
            trust_remote_code=True,
        ),
        generation_kwargs = {'do_sample': True},
        max_out_len=512,
        max_seq_len=4096,
        batch_size=8,
        meta_template=internlm1_chat_template,
        run_cfg=dict(num_gpus=1, num_procs=1),
        end_str='<eoa>',
    )
]


## ------------- Evaluation Configuration----------------
eval = dict(
    partitioner=dict(
        type=SubjectiveNaivePartitioner,
        mode='singlescore',
        models = models,
        judge_models = judge_models,
    ),
    runner=dict(
        type=LocalRunner,
        max_num_workers=256,
        task=dict(
            type=SubjectiveEvalTask
        )),
)

summarizer = dict(
    type=FlamesSummarizer, judge_type = 'general'
)

work_dir = 'outputs/flames/'
