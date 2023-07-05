from mmengine.config import read_base

with read_base():
    from .SuperGLUE_WSC_gen_d8d441 import WSC_datasets  # noqa: F401, F403
