from mmengine.config import read_base

with read_base():
    from .SuperGLUE_RTE_ppl_66caf3 import RTE_datasets  # noqa: F401, F403
