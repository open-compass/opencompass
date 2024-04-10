from mmengine.config import read_base

with read_base():
    from .datasets.ChemBench.ChemBench_gen import chembench_datasets
    from .models.mistral.hf_mistral_7b_instruct_v0_2 import models

datasets = [*chembench_datasets]
models = [*models]
