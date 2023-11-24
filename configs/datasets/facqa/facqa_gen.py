from mmengine.config import read_base

with read_base():
    from .facqa_gen_ds2r import facqa_datasets  # noqa: F401, F403