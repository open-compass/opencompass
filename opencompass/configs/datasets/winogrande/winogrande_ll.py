from mmengine.config import read_base

with read_base():
    from .winogrande_ll_c5cf57 import winogrande_datasets  # noqa: F401, F403
