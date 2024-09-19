from mmengine.config import read_base

with read_base():
    from opencompass.configs.datasets.demo.demo_gsm8k_chat_gen import gsm8k_datasets
    from opencompass.configs.models.hf_internlm.lmdeploy_internlm2_5_1_8b_chat import models

datasets = gsm8k_datasets
models = models
