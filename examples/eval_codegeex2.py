from mmengine.config import read_base

with read_base():
    from opencompass.configs.datasets.humanevalx.humanevalx_gen import \
        humanevalx_datasets
    from opencompass.configs.models.codegeex2.hf_codegeex2_6b import models

datasets = humanevalx_datasets
