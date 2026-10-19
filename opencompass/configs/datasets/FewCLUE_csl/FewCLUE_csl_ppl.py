from mmengine.config import read_base

with read_base():
    from .FewCLUE_csl_ppl_841b62 import csl_datasets  # noqa: F401, F403
