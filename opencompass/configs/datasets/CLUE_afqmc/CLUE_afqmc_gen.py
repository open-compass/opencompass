from mmengine.config import read_base

with read_base():
    from .CLUE_afqmc_gen_901306 import afqmc_datasets  # noqa: F401, F403
