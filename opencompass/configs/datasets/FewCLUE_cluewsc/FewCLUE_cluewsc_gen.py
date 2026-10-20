from mmengine.config import read_base

with read_base():
    from .FewCLUE_cluewsc_gen_c68933 import cluewsc_datasets  # noqa: F401, F403
