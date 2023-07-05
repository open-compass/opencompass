from mmengine.config import read_base

with read_base():
    from .winograd_ppl_c1c427 import winograd_datasets  # noqa: F401, F403
