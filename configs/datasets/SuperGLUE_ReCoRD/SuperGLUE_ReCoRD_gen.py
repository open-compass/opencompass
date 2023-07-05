from mmengine.config import read_base

with read_base():
    from .SuperGLUE_ReCoRD_gen_30dea0 import ReCoRD_datasets  # noqa: F401, F403
