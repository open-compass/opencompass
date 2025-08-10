from mmengine.config import read_base

with read_base():
    from .CLUE_ocnli_ppl_fdc6de import ocnli_datasets  # noqa: F401, F403
