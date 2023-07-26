from mmengine.config import read_base

with read_base():
    from .adv_mnli_gen_e6b269 import adv_mnli_datasets  # noqa: F401, F403
    from .adv_mnli_mm_gen_f98298 import adv_mnli_mm_datasets  # noqa: F401, F403
    from .adv_qnli_gen_c14d8d import adv_qnli_datasets  # noqa: F401, F403
    from .adv_qqp_gen_288920 import adv_qqp_datasets  # noqa: F401, F403
    from .adv_rte_gen_66dede import adv_rte_datasets  # noqa: F401, F403
    from .adv_sst2_gen_e1b9be import adv_sst2_datasets  # noqa: F401, F403

adv_glue_datasets = sum(
    (v for k, v in locals().items() if k.endswith('_datasets')), [])
