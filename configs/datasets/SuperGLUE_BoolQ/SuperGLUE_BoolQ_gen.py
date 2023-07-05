from mmengine.config import read_base

with read_base():
    from .SuperGLUE_BoolQ_gen_883d50 import BoolQ_datasets  # noqa: F401, F403
