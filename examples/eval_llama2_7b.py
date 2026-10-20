from mmengine.config import read_base

with read_base():
    from opencompass.configs.datasets.collections.base_medium_llama import (
        piqa_datasets, siqa_datasets)
    from opencompass.configs.models.llama.llama2_7b import models

datasets = [*piqa_datasets, *siqa_datasets]
