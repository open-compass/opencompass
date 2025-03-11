from mmengine import read_base

with read_base():
    from opencompass.configs.datasets.supergpqa.supergpqa_gen import \
        supergpqa_datasets 
    from opencompass.configs.models.qwen2_5.lmdeploy_qwen2_5_7b_instruct import \
        models 

datasets = supergpqa_datasets
models = models
