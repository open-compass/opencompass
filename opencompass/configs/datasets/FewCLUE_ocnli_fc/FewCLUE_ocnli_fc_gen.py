from mmengine.config import read_base

with read_base():
    from .FewCLUE_ocnli_fc_gen_f97a97 import ocnli_fc_datasets  # noqa: F401, F403
