from mmengine.config import read_base

with read_base():
    from .humaneval_pro_gen_ import humanevalpro_datasets  # noqa: F401, F403
