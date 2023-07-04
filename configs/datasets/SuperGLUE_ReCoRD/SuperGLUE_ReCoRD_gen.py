from mmengine.config import read_base

with read_base():
    from .SuperGLUE_ReCoRD_gen_d8f19c import ReCoRD_datasets  # noqa: F401, F403
