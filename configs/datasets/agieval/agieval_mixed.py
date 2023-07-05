from mmengine.config import read_base

with read_base():
    from .agieval_mixed_2f14ad import agieval_datasets  # noqa: F401, F403
