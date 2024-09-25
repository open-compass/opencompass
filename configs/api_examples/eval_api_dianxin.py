from mmengine.config import read_base
from opencompass.models import DianXin
from opencompass.partitioners import NaivePartitioner
from opencompass.runners.local_api import LocalAPIRunner
from opencompass.tasks import OpenICLInferTask

with read_base():
    from ..summarizers.medium import summarizer
    from ..datasets.ceval.ceval_gen import ceval_datasets

datasets = [
    *ceval_datasets
]

models = [
    dict(
        abbr='DianXin',
        type=DianXin,
        path='DianXin',
        key='xxxxxx',
        apiKey='xxxxxx', 
        url='xxxxxx',
        query_per_second=10,
        max_out_len=2048,
        max_seq_len=2048,
        batch_size=1),
]

infer = dict(partitioner=dict(type=NaivePartitioner),
             runner=dict(
                 type=LocalAPIRunner,
                 max_num_workers=1,
                 concurrent_users=1,
                 task=dict(type=OpenICLInferTask)), )

work_dir = 'outputs/api_DianXin/'