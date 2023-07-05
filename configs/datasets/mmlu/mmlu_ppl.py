from mmengine.config import read_base

with read_base():
    from .mmlu_ppl_c6bbe6 import mmlu_datasets  # noqa: F401, F403
