from mmengine.config import read_base

with read_base():
    from .SuperGLUE_AX_b_ppl_6db806 import AX_b_datasets  # noqa: F401, F403
