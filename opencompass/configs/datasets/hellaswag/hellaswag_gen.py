from mmengine.config import read_base

with read_base():
    from .hellaswag_gen_6faab5 import hellaswag_datasets  # noqa: F401, F403
