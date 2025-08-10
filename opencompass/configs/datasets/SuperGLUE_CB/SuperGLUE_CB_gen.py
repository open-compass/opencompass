from mmengine.config import read_base

with read_base():
    from .SuperGLUE_CB_gen_854c6c import CB_datasets  # noqa: F401, F403
