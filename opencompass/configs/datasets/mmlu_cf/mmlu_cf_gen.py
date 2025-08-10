from mmengine.config import read_base

with read_base():
    from .mmlu_cf_gen_040615 import mmlu_cf_datasets  # noqa: F401, F403
