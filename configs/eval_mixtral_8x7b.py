from mmengine.config import read_base

with read_base():
    from .datasets.collections.base_medium_llama import piqa_datasets, siqa_datasets
    from .models.mixtral.mixtral_8x7b_32k import models


datasets = [*piqa_datasets, *siqa_datasets]
