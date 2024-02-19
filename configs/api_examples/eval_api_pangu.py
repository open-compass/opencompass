from mmengine.config import read_base
from opencompass.models import PanGu
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
        abbr='pangu',
        type=PanGu,
        path='pangu',
        access_key='xxxxxx',
        secret_key='xxxxxx',
        url = 'xxxxxx',
        # url of token sever, used for generate token, like "https://xxxxxx.myhuaweicloud.com/v3/auth/tokens",
        token_url = 'xxxxxx',
        # scope-project-name, used for generate token
        project_name = 'xxxxxx',
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

work_dir = 'outputs/api_pangu/'
