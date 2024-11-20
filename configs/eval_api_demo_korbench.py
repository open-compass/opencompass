from mmengine.config import read_base

with read_base():
    #from opencompass.configs.datasets.demo.demo_gsm8k_chat_gen import gsm8k_datasets
    #from opencompass.configs.datasets.demo.demo_math_chat_gen import math_datasets
    from opencompass.configs.datasets.demo.demo_korbench_chat_gen import datasets as korbench_datasets
    from opencompass.configs.models.openai.gpt_4o_mini import models as gpt4
#datasets = gsm8k_datasets + korbench_datasets
datasets = korbench_datasets
models = gpt4
