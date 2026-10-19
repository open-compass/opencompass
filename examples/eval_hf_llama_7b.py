from mmengine.config import read_base

with read_base():
    from opencompass.configs.datasets.collections.base_medium_llama import (
        piqa_datasets, siqa_datasets)
    from opencompass.configs.models.hf_llama.hf_llama_7b import models

datasets = [*piqa_datasets, *siqa_datasets]
