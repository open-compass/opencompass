from mmengine.config import read_base

with read_base():
    from .mmlu_llmjudge_gen_f4336b import mmlu_datasets  # noqa: F401, F403