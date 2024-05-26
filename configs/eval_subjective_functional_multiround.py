from opencompass.models import HuggingFaceCausalLM
from copy import deepcopy
from opencompass.models import TurboMindModel
from mmengine.config import read_base

from opencompass.models import HuggingFaceCausalLM, HuggingFace, HuggingFaceChatGLM3, OpenAI
from opencompass.partitioners import NaivePartitioner, SizePartitioner
from opencompass.partitioners.sub_naive import SubjectiveNaivePartitioner
from opencompass.partitioners.sub_size import SubjectiveSizePartitioner
from opencompass.runners import LocalRunner
from opencompass.runners import SlurmSequentialRunner
from opencompass.tasks import OpenICLInferTask
from opencompass.tasks.subjective_eval import SubjectiveEvalTask
from opencompass.summarizers import MultiroundSummarizer

with read_base():
    from .datasets.subjective.multiround.functionalmt_zh_judgeby_gpt4 import subjective_datasets

api_meta_template = dict(
    round=[
        dict(role='HUMAN', api_role='HUMAN'),
        dict(role='BOT', api_role='BOT', generate=True),
    ]
)

_meta_template = dict(
    round=[
        dict(role='HUMAN', begin='<|im_start|>user\n', end='<|im_end|>\n'),
        dict(role='BOT', begin='<|im_start|>assistant\n', end='<|im_end|>\n', generate=True),
    ],
    eos_token_id=151645,
)

models = [
    dict(
        type=HuggingFaceCausalLM,
        abbr='qwen1.5-7b-chat-hf',
        path='Qwen/Qwen1.5-7B-Chat',
        model_kwargs=dict(
            device_map='auto',
            trust_remote_code=True
        ),
        tokenizer_kwargs=dict(
            padding_side='left',
            truncation_side='left',
            trust_remote_code=True,
            use_fast=False,
        ),
        generation_kwargs=dict(
            do_sample=True,
        ),
        meta_template=_meta_template,
        pad_token_id=151645,
        max_out_len=100,
        max_seq_len=2048,
        batch_size=8,
        run_cfg=dict(num_gpus=1, num_procs=1),
        end_str='<|im_end|>',
    )
]

datasets = [*subjective_datasets]

work_dir = 'outputs/multiround/'
# -------------Inferen Stage ----------------------------------------


infer = dict(
    partitioner=dict(type=SizePartitioner, max_task_size=1000),
    runner=dict(
        type=SlurmSequentialRunner,
        partition='your part',
        quotatype='auto',
        max_num_workers=256,
        task=dict(type=OpenICLInferTask)),
)

judge_models = [dict(
    abbr='GPT4-Turbo',
    type=OpenAI,
    path='gpt-4-1106-preview',
    key='',
        meta_template=api_meta_template,
        query_per_second=1,
        max_out_len=1024,
        max_seq_len=4096,
        batch_size=10,
        retry=10,
        temperature = 0,
)]

## ------------- Evaluation Configuration
eval = dict(
    partitioner=dict(
        type=SubjectiveSizePartitioner,
        max_task_size=1000,
        mode='singlescore',
        models = models,
        judge_models=judge_models
    ),
    runner=dict(
        type=SlurmSequentialRunner,
        partition='your part',
        quotatype='auto',
        max_num_workers=256,
        task=dict(type=SubjectiveEvalTask)),
)

summarizer = dict(
    type=MultiroundSummarizer
)
