from mmengine.config import read_base

with read_base():
    from .lveval_factrecall_zh_gen_dbee70 import (
        LVEval_factrecall_zh_datasets,
    )  # noqa: F401, F403
