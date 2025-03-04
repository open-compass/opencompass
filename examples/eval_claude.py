from mmengine.config import read_base

from opencompass.partitioners import NaivePartitioner
from opencompass.runners import LocalRunner
from opencompass.tasks import OpenICLInferTask

with read_base():
    # choose a list of datasets
    from opencompass.configs.datasets.collections.chat_medium import datasets
    from opencompass.configs.models.claude.claude import models
    # and output the results in a choosen format
    from opencompass.configs.summarizers.medium import summarizer

infer = dict(
    partitioner=dict(type=NaivePartitioner),
    runner=dict(type=LocalRunner,
                max_num_workers=8,
                task=dict(type=OpenICLInferTask)),
)
