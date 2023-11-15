from mmengine.config import read_base

with read_base():
    from .xquad_gen_3jk32 import xquad_datasets  # noqa: F401, F403