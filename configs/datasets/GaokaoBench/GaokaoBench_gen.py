from mmengine.config import read_base

with read_base():
    from .GaokaoBench_gen_aed980 import GaokaoBench_datasets  # noqa: F401, F403
