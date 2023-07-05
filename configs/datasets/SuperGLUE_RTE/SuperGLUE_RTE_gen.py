from mmengine.config import read_base

with read_base():
    from .SuperGLUE_RTE_gen_ce346a import RTE_datasets  # noqa: F401, F403
