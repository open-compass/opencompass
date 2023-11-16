from mmengine.config import read_base

with read_base():
    from .cmb_gen_dfb5c4 import cmb_datasets  # noqa: F401, F403
