from mmengine.config import read_base

with read_base():
    from .SuperGLUE_AX_g_ppl_66caf3 import AX_g_datasets  # noqa: F401, F403
