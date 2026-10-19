from mmengine.config import read_base

with read_base():
    from .adv_glue_qqp_gen_cdc277 import adv_qqp_datasets  # noqa: F401, F403
