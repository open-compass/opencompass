from mmengine.config import read_base
from opencompass.models import MiniMax
from opencompass.partitioners import NaivePartitioner
from opencompass.runners.local_api import LocalAPIRunner
from opencompass.tasks import OpenICLInferTask

with read_base():
    from ..summarizers.medium import summarizer
    from ..datasets.ceval.ceval_gen import ceval_datasets

datasets = [
    *ceval_datasets,
]

models = [
    dict(
        abbr='minimax_abab5.5-chat',
        type=MiniMax,
        path='abab5.5-chat',
        key='xxxxxxx', # please give you key
        group_id='xxxxxxxx', # please give your group_id
        query_per_second=1,
        max_out_len=2048,
        max_seq_len=2048,
        batch_size=8),
]

infer = dict(
    partitioner=dict(type=NaivePartitioner),
    runner=dict(
        type=LocalAPIRunner,
        max_num_workers=4,
        concurrent_users=4,
        task=dict(type=OpenICLInferTask)),
)

work_dir = 'outputs/api_minimax/'
