from mmengine.config import read_base

with read_base():
    from opencompass.configs.datasets.demo.demo_gsm8k_chat_gen import \
        gsm8k_datasets
    from opencompass.configs.datasets.demo.demo_math_chat_gen import \
        math_datasets
    from opencompass.configs.models.qwen3.vllm_qwen3_5_35b_a3b import \
        models as vllm_qwen3_5_35b_a3b_models
    from opencompass.configs.models.qwen3.vllm_qwen3_6_35b_a3b import \
        models as vllm_qwen3_6_35b_a3b_models

datasets = gsm8k_datasets + math_datasets
models = vllm_qwen3_5_35b_a3b_models + vllm_qwen3_6_35b_a3b_models
