from mmengine.config import read_base

with read_base():
    from .SuperGLUE_MultiRC_gen_27071f import MultiRC_datasets  # noqa: F401, F403
