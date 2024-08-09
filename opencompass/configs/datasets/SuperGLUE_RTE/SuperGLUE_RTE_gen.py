from mmengine.config import read_base

with read_base():
    from .SuperGLUE_RTE_gen_68aac7 import RTE_datasets  # noqa: F401, F403
