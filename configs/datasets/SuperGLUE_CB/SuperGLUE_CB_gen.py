from mmengine.config import read_base

with read_base():
    from .SuperGLUE_CB_gen_bb97e1 import CB_datasets  # noqa: F401, F403
