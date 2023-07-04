from mmengine.config import read_base

with read_base():
    from .SuperGLUE_CB_ppl_32adbb import CB_datasets  # noqa: F401, F403
