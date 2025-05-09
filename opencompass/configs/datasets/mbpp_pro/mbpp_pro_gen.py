from mmengine.config import read_base

with read_base():
    from .mbpp_pro_gen_ import mbpppro_datasets  # noqa: F401, F403
