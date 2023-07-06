from mmengine.config import read_base

with read_base():
    from .winogrande_ppl_55a66e import winogrande_datasets  # noqa: F401, F403
