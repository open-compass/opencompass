from mmengine.config import read_base

with read_base():
    from .medbench_mixed_2f14ad import medbench_datasets  # noqa: F401, F403
