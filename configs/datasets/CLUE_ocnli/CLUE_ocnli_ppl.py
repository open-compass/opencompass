from mmengine.config import read_base

with read_base():
    from .CLUE_ocnli_ppl_f103ab import ocnli_datasets  # noqa: F401, F403
