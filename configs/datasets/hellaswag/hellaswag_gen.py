from mmengine.config import read_base

with read_base():
    from .hellaswag_gen_cae9cb import hellaswag_datasets  # noqa: F401, F403
