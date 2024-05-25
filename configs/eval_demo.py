from mmengine.config import read_base

with read_base():
    from .datasets.siqa.siqa_gen import siqa_datasets
    from .datasets.winograd.winograd_ppl import winograd_datasets
    from .models.opt.hf_opt_125m import models as opt125m
    from .models.opt.hf_opt_350m import models as opt350m

datasets = [*siqa_datasets, *winograd_datasets]
models = [*opt125m, *opt350m]
