from mmengine.config import read_base

with read_base():
    from .cmmlu_ppl_3de98f import cmmlu_datasets  # noqa: F401, F403
