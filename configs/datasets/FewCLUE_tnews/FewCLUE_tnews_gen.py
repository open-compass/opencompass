from mmengine.config import read_base

with read_base():
    from .FewCLUE_tnews_gen_8d59ba import tnews_datasets  # noqa: F401, F403
