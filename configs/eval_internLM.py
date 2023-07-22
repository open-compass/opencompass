from mmengine.config import read_base

from opencompass.models import internLM
from opencompass.partitioners import SizePartitioner, NaivePartitioner
from opencompass.runners import DLCRunner, LocalRunner
from opencompass.tasks import OpenICLInferTask, OpenICLEvalTask

with read_base():
    from .datasets.nq.nq_gen import nq_datasets
    from .datasets.gsm8k.gsm8k_gen import gsm8k_datasets
datasets = [*nq_datasets]
# datasets += piqa_datasets
# datasets += nq_datasets

models = [
    dict(
        type=internLM,
        path="./internData/",
        tokenizer_path='./internData/V7.model',
        model_config="./internData/model_config.py",
        max_out_len=100,
        max_seq_len=2048,
        batch_size=16,
        run_cfg=dict(num_gpus=1, num_procs=1))
]
work_dir = './outputs/2023_07_20_02/'
infer = dict(
    partitioner=dict(type=SizePartitioner, max_task_size=2000, gen_task_coef=10),
    # partitioner=dict(type='NaivePartitioner'),
    runner=dict(
        type=LocalRunner,
        max_num_workers=32,
        task=dict(type=OpenICLInferTask),
        ),
)

eval = dict(
    partitioner=dict(type=NaivePartitioner),
    runner=dict(
        type=LocalRunner,
        max_num_workers=64,
        task=dict(type=OpenICLEvalTask),
        ),
)
