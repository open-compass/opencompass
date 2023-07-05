from mmengine.config import read_base

with read_base():
    from .mmlu_gen_a568f1 import mmlu_datasets  # noqa: F401, F403
