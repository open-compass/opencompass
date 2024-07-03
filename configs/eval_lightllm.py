from mmengine.config import read_base
from opencompass.models import LightllmAPI
from opencompass.partitioners import NaivePartitioner
from opencompass.runners import LocalRunner
from opencompass.tasks import OpenICLInferTask

with read_base():
    from .summarizers.leaderboard import summarizer
    from .datasets.humaneval.deprecated_humaneval_gen_a82cae import humaneval_datasets

datasets = [*humaneval_datasets]

'''
# Prompt template for InternLM2-Chat
# https://github.com/InternLM/InternLM/blob/main/chat/chat_format.md

_meta_template = dict(
    begin='<|im_start|>system\nYou are InternLM2-Chat, a harmless AI assistant<|im_end|>\n',
    round=[
        dict(role='HUMAN', begin='<|im_start|>user\n', end='<|im_end|>\n'),
        dict(role='BOT', begin='<|im_start|>assistant\n', end='<|im_end|>\n', generate=True),
    ]
)
'''

_meta_template = None

models = [
    dict(
        abbr='LightllmAPI',
        type=LightllmAPI,
        url='http://localhost:1030/generate',
        meta_template=_meta_template,
        batch_size=32,
        max_workers_per_task=128,
        rate_per_worker=1024,
        retry=4,
        generation_kwargs=dict(
            do_sample=False,
            ignore_eos=False,
            max_new_tokens=1024
        ),
    ),
]

infer = dict(
    partitioner=dict(type=NaivePartitioner),
    runner=dict(
        type=LocalRunner,
        max_num_workers=32,
        task=dict(type=OpenICLInferTask),
    ),
)
