from mmengine.config import read_base

from opencompass.models.openai_api import OpenAI
from opencompass.openicl import ChatInferencer
from opencompass.partitioners import SizePartitioner
from opencompass.runners import LocalRunner
from opencompass.tasks import OpenICLInferTask

with read_base():
    from .datasets.gsm8k.gsm8k_gen_1d7fe4 import gsm8k_datasets as datasets

models = [
    dict(
        abbr='gpt-3.5',
        type=OpenAI,
        path='gpt-3.5-turbo',
        key='ENV',
        max_out_len=100,
        max_seq_len=2048,
        batch_size=16,
        run_cfg=dict(num_gpus=1, num_procs=1),
    )
]

for dataset in datasets:
    # Use ChatInferencer instead of GenInferencer
    dataset['infer_cfg']['inferencer'] = dict(type=ChatInferencer)

infer = dict(
    partitioner=dict(type=SizePartitioner, max_task_size=1000),
    runner=dict(
        type=LocalRunner,
        max_num_workers=16,
        task=dict(type=OpenICLInferTask)),
)
