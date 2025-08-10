from mmengine.config import read_base

with read_base():
    from .korbench_single_0_shot_gen import korbench_0shot_single_datasets  # noqa: F401, F403