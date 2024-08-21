from mmengine.config import read_base

with read_base():
    from .sycophancy_gen_4bba45 import sycophancy_datasets  # noqa: F401, F403
