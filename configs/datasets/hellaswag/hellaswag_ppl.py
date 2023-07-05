from mmengine.config import read_base

with read_base():
    from .hellaswag_ppl_8e07d6 import hellaswag_datasets  # noqa: F401, F403
