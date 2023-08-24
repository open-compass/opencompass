from mmengine.config import read_base

with read_base():
    from .adv_glue_sst2.adv_glue_sst2_gen import adv_sst2_datasets
    from .adv_glue_qqp.adv_glue_qqp_gen import adv_qqp_datasets
    from .adv_glue_rte.adv_glue_rte_gen import adv_rte_datasets
    from .adv_glue_qnli.adv_glue_qnli_gen import adv_qnli_datasets
    from .adv_glue_mnli.adv_glue_mnli_gen import adv_mnli_datasets
    from .adv_glue_mnli_mm.adv_glue_mnli_mm_gen import adv_mnli_mm_datasets

datasets = sum((v for k, v in locals().items() if k.endswith('_datasets')), [])
