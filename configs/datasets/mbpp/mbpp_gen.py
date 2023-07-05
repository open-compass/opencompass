from mmengine.config import read_base

with read_base():
    from .mbpp_gen_4104e4 import mbpp_datasets  # noqa: F401, F403
