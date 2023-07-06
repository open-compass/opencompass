from mmengine.config import read_base

with read_base():
    from .FewCLUE_csl_gen_28b223 import csl_datasets  # noqa: F401, F403
