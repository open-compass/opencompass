from mmengine.config import read_base

with read_base():
    from .leval_quality_gen_36a006 import LEval_quality_datasets  # noqa: F401, F403
