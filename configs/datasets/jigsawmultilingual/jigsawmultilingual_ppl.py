from mmengine.config import read_base

with read_base():
    from .jigsawmultilingual_ppl_640128 import jigsawmultilingual_datasets  # noqa: F401, F403
