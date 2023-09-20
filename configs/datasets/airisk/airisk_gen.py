from mmengine.config import read_base

with read_base():
    from .airisk_gen_30480f import airisk_datasets  # noqa: F401, F403
