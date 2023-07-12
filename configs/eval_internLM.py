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
        path="/cpfs01/shared/public/users/chenkeyu1/models/linglongta_v4_2/116999",
        tokenizer_path='/cpfs01/shared/public/tokenizers/V7.model',
        tokenizer_type='v7',
        model_config = "/cpfs01/user/chenkeyu1/pjeval/InternLM/configs/model_config.py",
        max_out_len=100,
        max_seq_len=2048,
        batch_size=16,
        run_cfg=dict(num_gpus=1, num_procs=1))
]




