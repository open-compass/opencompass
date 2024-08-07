from mmengine.config import read_base

with read_base():
    from .mmlu_ppl_ac766d import mmlu_datasets  # noqa: F401, F403
