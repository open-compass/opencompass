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
        max_out_len=1024,
        batch_size=8,
        generation_kwargs=dict(
            do_sample=False,
            ignore_eos=False,
        ),
    ),
]

infer = dict(
    partitioner=dict(type=NaivePartitioner),
    runner=dict(
        type=LocalRunner,
        max_num_workers=8,
        task=dict(type=OpenICLInferTask),
    ),
)
