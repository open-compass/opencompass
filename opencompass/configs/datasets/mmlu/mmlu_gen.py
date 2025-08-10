from mmengine.config import read_base

with read_base():
    from .mmlu_openai_simple_evals_gen_b618ea import mmlu_datasets  # noqa: F401, F403