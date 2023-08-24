from mmengine.config import read_base
from opencompass.models.claude_api import Claude
from opencompass.partitioners import NaivePartitioner
from opencompass.runners import LocalRunner
from opencompass.tasks import OpenICLInferTask

with read_base():
    # choose a list of datasets
    from .datasets.collections.chat_medium import datasets
    # and output the results in a choosen format
    from .summarizers.medium import summarizer

models = [
    dict(abbr='Claude2',
        type=Claude,
        path='claude-2',
        key='YOUR_CLAUDE_KEY',
        query_per_second=1,
        max_out_len=2048, max_seq_len=2048, batch_size=2),
]

infer = dict(
    partitioner=dict(type=NaivePartitioner),
    runner=dict(
        type=LocalRunner,
        max_num_workers=8,
        task=dict(type=OpenICLInferTask)),
)
