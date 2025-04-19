from mmengine.config import read_base

with read_base():
    from .datasets.CLUE_cmnli.CLUE_cmnli_gen import cmnli_datasets
    from .models.mistral.hf_mistral_7b_v0_1 import models


datasets = [*cmnli_datasets]
