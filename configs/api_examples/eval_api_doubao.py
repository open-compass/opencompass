from mmengine.config import read_base
from opencompass.models import Doubao
from opencompass.partitioners import NaivePartitioner
from opencompass.runners.local_api import LocalAPIRunner
from opencompass.tasks import OpenICLInferTask

with read_base():
    # from .datasets.collections.chat_medium import datasets
    from ..summarizers.medium import summarizer
    from ..datasets.ceval.ceval_gen import ceval_datasets

datasets = [
    *ceval_datasets,
]

models = [
    dict(
        abbr='Doubao-pro-128k',
        type=Doubao,
        path='ep-xxxxxx',
        accesskey='Your_AK',
        secretkey='Your_SK',
        generation_kwargs={
            'temperature': 0.1,
            'top_p': 0.9,
        },
        query_per_second=1,
        max_out_len=2048,
        max_seq_len=2048,
        batch_size=8),
]

infer = dict(partitioner=dict(type=NaivePartitioner),
             runner=dict(
                 type=LocalAPIRunner,
                 max_num_workers=2,
                 concurrent_users=2,
                 task=dict(type=OpenICLInferTask)), )

work_dir = 'outputs/api_doubao/'
