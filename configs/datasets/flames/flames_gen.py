from mmengine.config import read_base

with read_base():
    from .flames_gen_0425 import flames_datasets  # noqa: F401, F403
