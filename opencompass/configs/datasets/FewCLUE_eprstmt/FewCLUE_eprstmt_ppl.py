from mmengine.config import read_base

with read_base():
    from .FewCLUE_eprstmt_ppl_f1e631 import eprstmt_datasets  # noqa: F401, F403
