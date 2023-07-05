from mmengine.config import read_base

with read_base():
    from .FewCLUE_ocnli_fc_gen_bef37f import ocnli_fc_datasets  # noqa: F401, F403
