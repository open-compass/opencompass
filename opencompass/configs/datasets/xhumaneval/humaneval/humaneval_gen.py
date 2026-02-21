from mmengine.config import read_base

with read_base():
    from .humaneval_gen_8e312c import humaneval_datasets  # noqa: F401, F403
