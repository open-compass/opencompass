from mmengine.config import read_base

with read_base():
    from .datasets.collections.base_medium import datasets
    from .models.hf_internlm_7b import models
    from .summarizers.medium import summarizer
