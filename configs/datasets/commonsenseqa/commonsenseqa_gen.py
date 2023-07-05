from mmengine.config import read_base

with read_base():
    from .commonsenseqa_gen_a58dbd import commonsenseqa_datasets  # noqa: F401, F403
