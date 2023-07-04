from mmengine.config import read_base

with read_base():
    from .FewCLUE_csl_ppl_8eee08 import csl_datasets  # noqa: F401, F403
