from mmengine.config import read_base

with read_base():
    from .adv_glue_sst2_gen_ee8d3b import adv_sst2_datasets  # noqa: F401, F403
