from mmengine.config import read_base

with read_base():
    from .datasets.FewCLUE_chid.FewCLUE_chid_knowledge_gen_0a29a2 import chid_knowledge_datasets
    from .models.hf_opt_125m import opt125m
    from .models.hf_opt_350m import opt350m

datasets = [*chid_knowledge_datasets]
models = [opt125m, opt350m]
