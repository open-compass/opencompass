from mmengine.config import read_base

with read_base():
    from .lveval_multifieldqa_en_mixup_gen_d7ea36 import (
        LVEval_multifieldqa_en_mixup_datasets,
    )  # noqa: F401, F403
