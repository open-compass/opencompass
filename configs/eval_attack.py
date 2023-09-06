from mmengine.config import read_base
from opencompass.partitioners import NaivePartitioner
from opencompass.runners import SlurmRunner
from opencompass.tasks import OpenICLAttackTask

with read_base():
    # choose a list of datasets
    from .datasets.promptbench.promptbench_wnli_gen_12 import wnli_datasets
    from .models.hf_vicuna_7b import models

datasets = wnli_datasets

infer = dict(
    partitioner=dict(type=NaivePartitioner),
    runner=dict(
        type=SlurmRunner,
        max_num_workers=8,
        task=dict(type=OpenICLAttackTask),
        retry=0),
)
