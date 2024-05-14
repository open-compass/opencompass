from mmengine.config import read_base
from opencompass.models.xunfei_api import XunFei
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
        abbr='Spark-v1-1',
        type=XunFei,
        appid='xxxx',
        path='ws://spark-api.xf-yun.com/v1.1/chat',
        api_secret = 'xxxxxxx',
        api_key = 'xxxxxxx',
        query_per_second=1,
        max_out_len=2048,
        max_seq_len=2048,
        batch_size=8),
    dict(
        abbr='Spark-v3-1',
        type=XunFei,
        appid='xxxx',
        domain='generalv3',
        path='ws://spark-api.xf-yun.com/v3.1/chat',
        api_secret = 'xxxxxxxx',
        api_key = 'xxxxxxxxx',
        query_per_second=1,
        max_out_len=2048,
        max_seq_len=2048,
        batch_size=8),
]

infer = dict(
    partitioner=dict(type=NaivePartitioner),
    runner=dict(
        type=LocalAPIRunner,
        max_num_workers=2,
        concurrent_users=2,
        task=dict(type=OpenICLInferTask)),
)

work_dir = 'outputs/api_xunfei/'
