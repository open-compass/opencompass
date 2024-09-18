from mmengine.config import read_base

from opencompass.models import BaiLingAPI
from opencompass.partitioners import NaivePartitioner
from opencompass.runners.local_api import LocalAPIRunner
from opencompass.tasks import OpenICLInferTask

with read_base():
    from opencompass.configs.datasets.ceval.ceval_gen import ceval_datasets
    from opencompass.configs.summarizers.medium import summarizer

datasets = [
    *ceval_datasets,
]

models = [
    dict(
        path="Bailing-Lite-0830",
        token="xxxxxx",  # please give your key
        url="https://bailingchat.alipay.com/chat/completions",
        type=BaiLingAPI,
        generation_kwargs={},
        query_per_second=1,
        max_seq_len=2048,
    ),
]

infer = dict(
    partitioner=dict(type=NaivePartitioner),
    runner=dict(
        type=LocalAPIRunner,
        max_num_workers=2,
        concurrent_users=2,
        task=dict(type=OpenICLInferTask),
    ),
)

work_dir = "outputs/api_bailing/"
