from mmengine.config import read_base

with read_base():
    from .FewCLUE_cluewsc_ppl_2a9e61 import cluewsc_datasets  # noqa: F401, F403
