from mmengine.config import read_base

from opencompass.models import BailingAPI
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
        path='Bailing-Lite-1116',
        token='xxxxxx',  # set your key here or in environment variable BAILING_API_KEY
        url='https://bailingchat.alipay.com/chat/completions',
        type=BailingAPI,
        max_out_len=11264,
        batch_size=1,
        generation_kwargs={
            'temperature': 0.01,
            'top_p': 1.0,
            'top_k': -1,
            'n': 1,
            'logprobs': 1,
        },
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

work_dir = 'outputs/api_bailing/'
