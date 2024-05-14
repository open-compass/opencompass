from mmengine.config import read_base
from opencompass.models import ERNIEBot
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
        abbr='erniebot',
        type=ERNIEBot,
        path='erniebot',
        key='xxxxxx',  # please give you key
        secretkey='xxxxxxxxx',  # please give your group_id
        url='xxxxxxxxx',
        generation_kwargs = {
            'temperature': 0.8,
        },
        query_per_second=1,
        max_out_len=2048,
        max_seq_len=2048,
        batch_size=8
    ),
]

infer = dict(
    partitioner=dict(type=NaivePartitioner),
    runner=dict(
        type=LocalAPIRunner,
        max_num_workers=2,
        concurrent_users=2,
        task=dict(type=OpenICLInferTask)),
)

work_dir = 'outputs/api_erniebot/'
