from mmengine.config import read_base

with read_base():
    from .models.qwen.hf_qwen_7b_chat import models
    from .datasets.lawbench.lawbench_zero_shot_gen_002588 import lawbench_datasets as lawbench_zero_shot_datasets
    from .datasets.lawbench.lawbench_one_shot_gen_002588 import lawbench_datasets as lawbench_one_shot_datasets
    from .summarizers.lawbench import summarizer

datasets = lawbench_zero_shot_datasets + lawbench_one_shot_datasets
for d in datasets:
    d['infer_cfg']['inferencer']['save_every'] = 1
