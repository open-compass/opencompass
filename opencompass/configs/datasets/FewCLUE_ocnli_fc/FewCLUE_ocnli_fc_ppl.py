from mmengine.config import read_base

with read_base():
    from .FewCLUE_ocnli_fc_ppl_c08300 import ocnli_fc_datasets  # noqa: F401, F403
