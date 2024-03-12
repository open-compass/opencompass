from mmengine.config import read_base

with read_base():
    from .lveval_multifieldqa_zh_mixup_gen_0fbdad import (
        LVEval_multifieldqa_zh_mixup_datasets,
    )  # noqa: F401, F403
