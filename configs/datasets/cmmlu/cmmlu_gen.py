from mmengine.config import read_base

with read_base():
    from .cmmlu_gen_ffe7c0 import cmmlu_datasets  # noqa: F401, F403
