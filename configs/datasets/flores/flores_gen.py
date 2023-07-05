from mmengine.config import read_base

with read_base():
    from .flores_gen_8eb9ca import flores_datasets  # noqa: F401, F403
