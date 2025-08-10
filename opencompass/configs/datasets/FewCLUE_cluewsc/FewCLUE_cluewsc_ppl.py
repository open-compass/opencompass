from mmengine.config import read_base

with read_base():
    from .FewCLUE_cluewsc_ppl_868415 import cluewsc_datasets  # noqa: F401, F403
