from mmengine.config import read_base

with read_base():
    from .SuperGLUE_WSC_gen_8a881c import WSC_datasets  # noqa: F401, F403
