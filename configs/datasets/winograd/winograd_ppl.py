from mmengine.config import read_base

with read_base():
    from .winograd_ppl_b6c7ed import winograd_datasets  # noqa: F401, F403
