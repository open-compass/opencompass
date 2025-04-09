from mmengine.config import read_base

with read_base():
    from .cmmlu_0shot_cot_gen_305931 import cmmlu_datasets  # noqa: F401, F403