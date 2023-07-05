from mmengine.config import read_base

with read_base():
    from .winogrande_ppl_00f8ad import winogrande_datasets  # noqa: F401, F403
