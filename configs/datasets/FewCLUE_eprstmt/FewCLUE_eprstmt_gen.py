from mmengine.config import read_base

with read_base():
    from .FewCLUE_eprstmt_gen_d6d06d import eprstmt_datasets  # noqa: F401, F403
