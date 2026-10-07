from mmengine import read_base

with read_base():
    from opencompass.configs.datasets.reasonzoo.reasonzoo_single_0_shot_gen import \
        reasonzoo_0shot_single_datasets as zero_shot_datasets
    from opencompass.configs.models.openai.gpt_4o_2024_05_13 import models as gpt4o

datasets = zero_shot_datasets
models = gpt4o
