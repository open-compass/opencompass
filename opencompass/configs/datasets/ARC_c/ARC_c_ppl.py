from mmengine.config import read_base

with read_base():
    from .ARC_c_ppl_a450bd import ARC_c_datasets  # noqa: F401, F403
