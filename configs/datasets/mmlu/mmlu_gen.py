from mmengine.config import read_base

with read_base():
    from .mmlu_gen_4d595a import mmlu_datasets  # noqa: F401, F403
