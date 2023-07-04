from mmengine.config import read_base

with read_base():
    from .SuperGLUE_COPA_gen_6d5e67 import COPA_datasets  # noqa: F401, F403
