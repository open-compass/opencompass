from mmengine.config import read_base

with read_base():
    from .cmb_ppl_xxxxxx import cmb_datasets  # noqa: F401, F403
