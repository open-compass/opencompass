from mmengine.config import read_base

with read_base():
    from .squad20_gen_1710bc import squad20_datasets  # noqa: F401, F403
