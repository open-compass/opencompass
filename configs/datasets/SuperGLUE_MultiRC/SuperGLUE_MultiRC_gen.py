from mmengine.config import read_base

with read_base():
    from .SuperGLUE_MultiRC_gen_26c9dc import MultiRC_datasets  # noqa: F401, F403
