from mmengine.config import read_base
from opencompass.models import BaiChuan
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
        abbr='Baichuan2-53B',
        type=BaiChuan,
        path='Baichuan2-53B',
        api_key='xxxxxx',
        secret_key='xxxxx',
        url='xxxxx',
        generation_kwargs={
            'temperature': 0.3,
            'top_p': 0.85,
            'top_k': 5,
            'with_search_enhance': False,
        },
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

work_dir = 'outputs/api_baichuan53b/'
