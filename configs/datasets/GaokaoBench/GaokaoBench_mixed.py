from mmengine.config import read_base

with read_base():
    from .GaokaoBench_mixed_f2038e import GaokaoBench_datasets  # noqa: F401, F403
