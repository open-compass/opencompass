from mmengine.config import read_base

with read_base():
    from .mmmlu_5_shot_b31abe import mmmlu_datasets  # noqa: F401, F403