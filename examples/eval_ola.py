from mmengine.config import read_base

with read_base():
    from opencompass.configs.datasets.demo.demo_gsm8k_chat_gen import gsm8k_datasets
    from opencompass.configs.datasets.demo.demo_math_chat_gen import math_datasets
    from opencompass.configs.models.ola.ola import models as ola_models


datasets = math_datasets
models = ola_models