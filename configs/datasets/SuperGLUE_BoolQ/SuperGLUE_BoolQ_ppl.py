from mmengine.config import read_base

with read_base():
    from .SuperGLUE_BoolQ_ppl_f80fb0 import BoolQ_datasets  # noqa: F401, F403
