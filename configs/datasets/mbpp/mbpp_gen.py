from mmengine.config import read_base

with read_base():
    from .mbpp_gen_1e1056 import mbpp_datasets  # noqa: F401, F403
