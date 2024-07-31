from mmengine.config import read_base

with read_base():
    from .datasets.SciCode.SciCode_gen import SciCode_datasets
    from .models.qwen.hf_qwen2_1_5b_instruct import models as hf_qwen2_1_5b_instruct_models

datasets = SciCode_datasets
models = hf_qwen2_1_5b_instruct_models
