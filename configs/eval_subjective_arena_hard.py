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
from opencompass.summarizers import ArenaHardSummarizer

with read_base():
    from .datasets.subjective.arena_hard.arena_hard_scoring import subjective_datasets

api_meta_template = dict(
    round=[
        dict(role='HUMAN', api_role='HUMAN'),
        dict(role='BOT', api_role='BOT', generate=True),
    ]
)

_meta_template = dict(
    round=[
        dict(role='HUMAN', begin='<|begin_of_text|>user<|end_header_id|>\n\n', end='<|eot_id|>'),
        dict(role='BOT', begin='<|begin_of_text|>assistant<|end_header_id|>\n\n', end='<|eot_id|>', generate=True),
    ],
)

models = [
    dict(
        type=HuggingFaceCausalLM,
        abbr='llama-3-8b-instruct-hf',
        path='meta-llama/Meta-Llama-3-8B-Instruct',
        model_kwargs=dict(device_map='auto'),
        tokenizer_kwargs=dict(
            padding_side='left',
            truncation_side='left',
            use_fast=False,
        ),
        meta_template=_meta_template,
        max_out_len=4096,
        max_seq_len=2048,
        batch_size=8,
        run_cfg=dict(num_gpus=1, num_procs=1),
        generation_kwargs={'eos_token_id': [128001, 128009]},
        batch_padding=True,
    )
]

datasets = [*subjective_datasets]

work_dir = 'outputs/arena_hard/'
# -------------Inferen Stage ----------------------------------------


infer = dict(
    partitioner=dict(type=SizePartitioner, max_task_size=1000000),
    runner=dict(
        type=LocalRunner,
        max_num_workers=32,
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
gpt4_0314 = dict(
    abbr='gpt4-0314',
    type=OpenAI,
)

eval = dict(
    partitioner=dict(
        type=SubjectiveSizePartitioner,
        max_task_size=1000000,
        mode='m2n',
        infer_order='double',
        base_models=[gpt4_0314],
        compare_models=models,
        judge_models=judge_models,
    ),
    runner=dict(type=LocalRunner, max_num_workers=16, task=dict(type=SubjectiveEvalTask)),
    given_pred = [{'abbr':'gpt4-0314', 'path':''}]
)

summarizer = dict(
    type=ArenaHardSummarizer
)
