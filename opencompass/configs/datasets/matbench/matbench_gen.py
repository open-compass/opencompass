from mmengine.config import read_base

with read_base():
    # from .matbench_gen_regex_judge import matbench_datasets  # noqa: F401, F403
    from .matbench_llm_judge_gen_0e9276 import matbench_datasets  # noqa: F401, F403
