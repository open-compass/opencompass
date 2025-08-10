from mmengine.config import read_base

with read_base():
    from .SuperGLUE_MultiRC_ppl_ced824 import MultiRC_datasets  # noqa: F401, F403
