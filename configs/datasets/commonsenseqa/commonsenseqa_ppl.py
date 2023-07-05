from mmengine.config import read_base

with read_base():
    from .commonsenseqa_ppl_2ca33c import commonsenseqa_datasets  # noqa: F401, F403
