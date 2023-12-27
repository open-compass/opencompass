from mmengine.config import read_base
from opencompass.models import LightllmAPI
from opencompass.partitioners import NaivePartitioner
from opencompass.runners import LocalRunner
from opencompass.tasks import OpenICLInferTask

with read_base():
    from .datasets.humaneval.humaneval_gen import humaneval_datasets

datasets = [*humaneval_datasets]

models = [
    dict(
        abbr='LightllmAPI',
        type=LightllmAPI,
        url='http://localhost:8080/generate',
        max_seq_len=2048,
        batch_size=32,
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
