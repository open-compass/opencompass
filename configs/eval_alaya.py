from mmengine.config import read_base

with read_base():
    from .datasets.ceval.ceval_gen import ceval_datasets
    from .datasets.cmmlu.cmmlu_gen import cmmlu_datasets
    from .datasets.agieval.agieval_gen import agieval_datasets
    from .datasets.bbh.bbh_gen import bbh_datasets
    from .datasets.mmlu.mmlu_gen import mmlu_datasets
    from .models.alaya.alaya import models

datasets = [*bbh_datasets, *ceval_datasets, *cmmlu_datasets, *agieval_datasets, *mmlu_datasets]
