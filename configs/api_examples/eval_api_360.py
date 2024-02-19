from mmengine.config import read_base
from opencompass.models import AI360GPT
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
        abbr='360GPT_S2_V9',
        type=AI360GPT,
        path='360GPT_S2_V9',
        key='xxxxxxxxxxxx',
        generation_kwargs={
            'temperature': 0.9,
            'max_tokens': 2048,
            'top_p': 0.5,
            'tok_k': 0,
            'repetition_penalty': 1.05,
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

work_dir ='./output/api_360GPT_S2_V9'
