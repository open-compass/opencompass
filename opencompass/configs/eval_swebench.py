from mmengine.config import read_base

with read_base():
    from opencompass.configs.datasets.swebench.swebench_gen import swebench_datasets
    from opencompass.configs.models.swe_llama.vllm_swe_llama_7b import models

datasets = swebench_datasets
