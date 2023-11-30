from mmengine.config import read_base

with read_base():
    from .cmmlu_gen_c13365 import cmmlu_datasets  # noqa: F401, F403
