from mmengine.config import read_base

with read_base():
    from .s3eval_gen_b8ac80 import s3eval_datasets  # noqa: F401, F40
