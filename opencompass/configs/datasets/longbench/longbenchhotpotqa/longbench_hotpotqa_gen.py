from mmengine.config import read_base

with read_base():
    from .longbench_hotpotqa_gen_6b3efc import LongBench_hotpotqa_datasets  # noqa: F401, F403
