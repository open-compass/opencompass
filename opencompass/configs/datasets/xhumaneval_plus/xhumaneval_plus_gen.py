from mmengine.config import read_base

with read_base():
    from .humaneval_plus_gen_8e312c import humaneval_plus_datasets  # noqa: F401, F403
