from mmengine.config import read_base

from opencompass.models.openai_api import OpenAI
from opencompass.partitioners import SizePartitioner
from opencompass.runners import LocalRunner
from opencompass.tasks import OpenICLInferTask

with read_base():
    from opencompass.configs.datasets.gsm8k.gsm8k_gen_d6de81 import \
        gsm8k_datasets
    from opencompass.configs.datasets.math.math_gen_1ed9c2 import math_datasets
    from opencompass.configs.datasets.MathBench.mathbench_gen import \
        mathbench_datasets
    from opencompass.configs.summarizers.math_baseline import summarizer

datasets = []
datasets += gsm8k_datasets
datasets += math_datasets
datasets += mathbench_datasets

models = [
    dict(
        abbr='gpt-3.5-react',
        type=OpenAI,
        path='gpt-3.5-turbo',
        key='ENV',
        query_per_second=1,
        max_seq_len=4096,
        batch_size=1,
    ),
]

infer = dict(
    partitioner=dict(type=SizePartitioner, max_task_size=1000),
    runner=dict(type=LocalRunner,
                max_num_workers=16,
                task=dict(type=OpenICLInferTask)),
)
