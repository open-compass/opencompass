from mmengine.config import read_base

with read_base():
    from .FewCLUE_bustm_gen_305431 import bustm_datasets  # noqa: F401, F403
