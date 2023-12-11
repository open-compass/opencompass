from mmengine.config import read_base

with read_base():
    from .winogrande_ppl_8be6c3 import winogrande_datasets  # noqa: F401, F403
