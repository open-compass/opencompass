from mmengine.config import read_base

with read_base():
    from .CLUE_ocnli_gen_c4cb6c import ocnli_datasets  # noqa: F401, F403
