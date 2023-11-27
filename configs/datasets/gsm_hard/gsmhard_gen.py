from mmengine.config import read_base

with read_base():
    from .gsmhard_gen_8a1400 import gsmhard_datasets  # noqa: F401, F403
