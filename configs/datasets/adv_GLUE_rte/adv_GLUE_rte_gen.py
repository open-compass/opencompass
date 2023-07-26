from mmengine.config import read_base

with read_base():
    from .adv_GLUE_rte_gen_8cc547 import adv_rte_datasets  # noqa: F401, F403
