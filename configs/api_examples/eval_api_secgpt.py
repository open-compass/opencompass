from mmengine.config import read_base
from opencompass.summarizers import MultiroundSummarizer

from opencompass.models import MyOpenAI
from opencompass.partitioners import NaivePartitioner
from opencompass.runners.local_api import LocalAPIRunner
from opencompass.tasks import OpenICLInferTask

with read_base():
    from ..summarizers.medium import summarizer
    from ..datasets.ceval.ceval_gen import ceval_datasets
    from ..datasets.gsm8k.gsm8k_gen_1d7fe4 import gsm8k_datasets


datasets = [
    *gsm8k_datasets,
]

models = [
    dict(
       # abbr='SecGPT',
        abbr='diaoyu',
        type=MyOpenAI,
        path='SecGPT',
        key='test',  # please give you key
        # generation_kwargs={
        #     'enable_search': False,
        # },
        query_per_second=1,
        max_out_len=2048,
        max_seq_len=2048,
        batch_size=16
    ),
]

infer = dict(
    partitioner=dict(type=NaivePartitioner),
    runner=dict(
        type=LocalAPIRunner,
        max_num_workers=1,
        concurrent_users=1,
        task=dict(type=OpenICLInferTask)),
)

work_dir = 'outputs/api_SecGPT/'
summarizer = dict(
    type=MultiroundSummarizer
)