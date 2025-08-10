from mmengine.config import read_base

with read_base():
    from .cmmlu_ppl_8b9c76 import cmmlu_datasets  # noqa: F401, F403
