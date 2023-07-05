from mmengine.config import read_base

with read_base():
    from .agieval_gen_dc7dae import agieval_datasets  # noqa: F401, F403
