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
        key='870C68A2E4744F07BB0647BC1206698A',
        url='https://150.223.245.42/csrobot/cschannels/openapi/evaluation/chat/dialog?apiKey=CA159B1FFFFA44C793B530843D8F6D12',
        query_per_second=10,
        max_out_len=2048,
        max_seq_len=2048,
        batch_size=8),
]

infer = dict(partitioner=dict(type=NaivePartitioner),
             runner=dict(
                 type=LocalAPIRunner,
                 max_num_workers=1,
                 concurrent_users=1,
                 task=dict(type=OpenICLInferTask)), )

work_dir = 'outputs/api_DianXin/'