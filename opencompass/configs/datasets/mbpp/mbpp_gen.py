from mmengine.config import read_base

with read_base():
    from .mbpp_gen_830460 import mbpp_datasets  # noqa: F401, F403
