from mmengine.config import read_base

with read_base():
    from .SuperGLUE_WSC_ppl_cbf31c import WSC_datasets  # noqa: F401, F403
