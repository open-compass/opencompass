from mmengine.config import read_base

with read_base():
    from .SuperGLUE_WSC_ppl_d0f531 import WSC_datasets  # noqa: F401, F403
