from opencompass.models.intern import intern_model
from mmengine.config import read_base

with read_base():
    from .datasets.piqa.piqa_ppl import piqa_datasets
    from .datasets.nq.nq_gen import nq_datasets
datasets = []
datasets += piqa_datasets
datasets += nq_datasets

models = [
    dict(
        type=intern_model,
        path="model/",
        tokenizer_path='V7.model',
        model_config = "model_config.py",
        max_out_len=100,
        max_seq_len=2048,
        batch_size=16,
        run_cfg=dict(num_gpus=1, num_procs=1))
]


