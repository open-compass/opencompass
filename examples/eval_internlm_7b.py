from mmengine.config import read_base

with read_base():
    # choose a list of datasets
    from opencompass.configs.datasets.collections.base_medium import datasets
    # choose a model of interest
    from opencompass.configs.models.hf_internlm.hf_internlm_7b import models
    # and output the results in a choosen format
    from opencompass.configs.summarizers.medium import summarizer
