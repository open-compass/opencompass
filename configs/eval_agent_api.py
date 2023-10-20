from mmengine.config import read_base
from opencompass.partitioners import SizePartitioner
from opencompass.runners import LocalRunner
from opencompass.tasks import OpenICLInferTask
from opencompass.models import OpenAI

with read_base():
    from .datasets.math.math_gen_66176f import math_datasets, math_example
    from .models.agent_template import model_template

datasets = math_datasets
models = [
    dict(
        abbr='gpt-3.5-react',
        llm=dict(
            type=OpenAI,
            path='gpt-3.5-turbo',
            key='ENV',
            query_per_second=1,
            max_seq_len=4096,
        ),
        **model_template,
        example=math_example,
        batch_size=8),
]

infer = dict(
    partitioner=dict(type=SizePartitioner, max_task_size=40000),
    runner=dict(
        type=LocalRunner, max_num_workers=16,
        task=dict(type=OpenICLInferTask)),
)
