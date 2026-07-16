from mmengine.config import read_base

with read_base():
    from .gpqa_gen_4baadb import gpqa_datasets  # noqa: F401, F403
