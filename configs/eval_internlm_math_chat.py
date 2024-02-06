from mmengine.config import read_base
from opencompass.models.huggingface import HuggingFaceCausalLM

with read_base():
    # choose a list of datasets
    from .datasets.gsm8k.gsm8k_gen import gsm8k_datasets
    from .datasets.math.math_gen_736506 import math_datasets

    from .models.hf_internlm.hf_internlm2_chat_math_7b import models as internlm_math_chat_7b_models
    from .models.hf_internlm.hf_internlm2_chat_math_20b import models as internlm_math_chat_20b_models

datasets = sum((v for k, v in locals().items() if k.endswith('_datasets')), [])
# Eval Math and GSM8k for both Internlm-Math-Chat-7B and 20b
datasets = [*math_datasets, *gsm8k_datasets]
models = [*internlm_math_chat_7b_models, *internlm_math_chat_20b_models]
