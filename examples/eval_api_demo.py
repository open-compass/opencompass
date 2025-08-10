from mmengine.config import read_base

with read_base():
    from opencompass.configs.datasets.demo.demo_gsm8k_chat_gen import \
        gsm8k_datasets
    from opencompass.configs.datasets.demo.demo_math_chat_gen import \
        math_datasets
    from opencompass.configs.models.openai.gpt_4o_2024_05_13 import \
        models as gpt4

datasets = gsm8k_datasets + math_datasets
models = gpt4
