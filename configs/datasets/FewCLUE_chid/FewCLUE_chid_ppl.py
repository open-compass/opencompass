from mmengine.config import read_base

with read_base():
    from .FewCLUE_chid_ppl_b6cd88 import chid_datasets  # noqa: F401, F403
