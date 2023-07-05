from mmengine.config import read_base

with read_base():
    from .triviaqarc_gen_db6413 import triviaqarc_datasets  # noqa: F401, F403
