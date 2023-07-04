from mmengine.config import read_base

with read_base():
    from .CLUE_cmnli_gen_316313 import cmnli_datasets  # noqa: F401, F403
