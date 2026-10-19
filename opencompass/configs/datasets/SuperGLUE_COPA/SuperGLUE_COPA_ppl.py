from mmengine.config import read_base

with read_base():
    from .SuperGLUE_COPA_ppl_9f3618 import COPA_datasets  # noqa: F401, F403
