from mmengine.config import read_base

with read_base():
    from .FewCLUE_tnews_gen_b90e4a import tnews_datasets  # noqa: F401, F403
