from mmengine.config import read_base

with read_base():
    from .mmlu_gen_a484b3 import mmlu_datasets  # noqa: F401, F403
