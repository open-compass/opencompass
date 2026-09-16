from mmengine.config import read_base

with read_base():
    from opencompass.configs.datasets.humanevalx.humanevalx_rawprompt_gen_386eb8 import \
        humanevalx_datasets
    from opencompass.configs.models.openai.gpt_6_astra import models

datasets = humanevalx_datasets
