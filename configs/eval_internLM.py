from mmengine.config import read_base

from opencompass.models import internLM

with read_base():
    from .datasets.nq.nq_gen import nq_datasets

datasets = [*nq_datasets]


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