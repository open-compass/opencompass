from mmengine.config import read_base

with read_base():
    from .FewCLUE_tnews_ppl_d10e8a import tnews_datasets  # noqa: F401, F403
