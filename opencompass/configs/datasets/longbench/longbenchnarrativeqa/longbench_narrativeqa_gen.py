from mmengine.config import read_base

with read_base():
    from .longbench_narrativeqa_gen_a68305 import LongBench_narrativeqa_datasets  # noqa: F401, F403
