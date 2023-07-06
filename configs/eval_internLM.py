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
        path="/mnt/petrelfs/chenkeyu1/program/model/final_maibao",
        tokenizer_path='/mnt/petrelfs/share_data/llm_data/tokenizers/V7.model',
        tokenizer_type='v7',
        model_config = "/mnt/petrelfs/chenkeyu1/program/openAGIEval/InternLM/configs/model_config.py",
        max_out_len=100,
        max_seq_len=2048,
        batch_size=16,
        run_cfg=dict(num_gpus=1, num_procs=1))
]




