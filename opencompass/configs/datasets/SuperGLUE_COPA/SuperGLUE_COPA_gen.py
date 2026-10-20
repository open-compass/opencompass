from mmengine.config import read_base

with read_base():
    from .SuperGLUE_COPA_gen_91ca53 import COPA_datasets  # noqa: F401, F403
