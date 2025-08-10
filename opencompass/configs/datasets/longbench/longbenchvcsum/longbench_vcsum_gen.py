from mmengine.config import read_base

with read_base():
    from .longbench_vcsum_gen_f7a8ac import LongBench_vcsum_datasets  # noqa: F401, F403
