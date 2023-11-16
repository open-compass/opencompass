from mmengine.config import read_base
from opencompass.models import AI360GPT
from opencompass.partitioners import NaivePartitioner
from opencompass.runners import LocalRunner
from opencompass.runners.local_api import LocalAPIRunner
from opencompass.tasks import OpenICLInferTask

with read_base():
    # from .datasets.collections.chat_medium import datasets
    from .summarizers.medium import summarizer
    from .datasets.ceval.ceval_gen import ceval_datasets
    # from .datasets.ARC_c.ARC_c_gen import ARC_c_datasets
    # from .datasets.race.race_gen import race_datasets
    # from .datasets.commonsenseqa.commonsenseqa_gen_260dab import commonsenseqa_datasets
    # from .datasets.winogrande.winogrande_gen import winogrande_datasets
    # from .datasets.gsm8k.gsm8k_gen import gsm8k_datasets

datasets = [
    *ceval_datasets,
    # *ARC_c_datasets,
    # *race_datasets,
    # *commonsenseqa_datasets,
    # *winogrande_datasets,
    # *gsm8k_datasets,
]

models = [
    dict(
        abbr='360GPT_S2_V9',
        type=AI360GPT,
        path='360GPT_S2_V9',
        key="xxxxxxxxxxxx",
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

work_dir ="./output/360GPT_S2_V9"