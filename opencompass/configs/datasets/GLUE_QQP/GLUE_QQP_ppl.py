from mmengine.config import read_base

with read_base():
    from .GLUE_QQP_ppl_250d00 import QQP_datasets  # noqa: F401, F403
