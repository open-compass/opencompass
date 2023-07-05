from mmengine.config import read_base

with read_base():
    from .obqa_ppl_2b5b12 import obqa_datasets  # noqa: F401, F403
