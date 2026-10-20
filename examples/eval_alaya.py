from mmengine.config import read_base

with read_base():
    from opencompass.configs.datasets.agieval.agieval_gen import \
        agieval_datasets
    from opencompass.configs.datasets.bbh.bbh_gen import bbh_datasets
    from opencompass.configs.datasets.ceval.ceval_gen import ceval_datasets
    from opencompass.configs.datasets.cmmlu.cmmlu_gen import cmmlu_datasets
    from opencompass.configs.datasets.mmlu.mmlu_gen import mmlu_datasets
    from opencompass.configs.models.alaya.alaya import models

datasets = [
    *bbh_datasets, *ceval_datasets, *cmmlu_datasets, *agieval_datasets,
    *mmlu_datasets
]
