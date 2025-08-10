from mmengine.config import read_base

from opencompass.partitioners import NaivePartitioner
from opencompass.runners import LocalRunner
from opencompass.tasks import OpenICLAttackTask

with read_base():
    # choose a list of datasets
    from opencompass.configs.datasets.promptbench.promptbench_wnli_gen_50662f import \
        wnli_datasets
    from opencompass.configs.models.qwen.hf_qwen2_1_5b import models

datasets = wnli_datasets

# Please run whole dataset at a time, aka use `NaivePartitioner` only
# Please use `OpenICLAttackTask` if want to perform attack experiment
infer = dict(
    partitioner=dict(type=NaivePartitioner),
    runner=dict(type=LocalRunner,
                max_num_workers=8,
                task=dict(type=OpenICLAttackTask)),
)

attack = dict(
    attack='textfooler',
    query_budget=100,
    prompt_topk=1,
)
