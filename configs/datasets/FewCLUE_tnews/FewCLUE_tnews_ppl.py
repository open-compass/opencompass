from mmengine.config import read_base

with read_base():
    from .FewCLUE_tnews_ppl_784b9e import tnews_datasets  # noqa: F401, F403
