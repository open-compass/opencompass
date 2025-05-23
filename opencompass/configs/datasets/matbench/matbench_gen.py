from mmengine.config import read_base

with read_base():
    # from .matbench_gen_testing_updated import matbench_datasets  # noqa: F401, F403
    from .matbench_gen_llm_judge import matbench_datasets  # noqa: F401, F403
