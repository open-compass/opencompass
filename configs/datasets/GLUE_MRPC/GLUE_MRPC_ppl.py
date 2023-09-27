from mmengine.config import read_base

with read_base():
    from .GLUE_MRPC_ppl_96564c import MRPC_datasets  # noqa: F401, F403
