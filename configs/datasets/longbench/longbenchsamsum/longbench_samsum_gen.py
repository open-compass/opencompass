from mmengine.config import read_base

with read_base():
    from .longbench_samsum_gen_f4416d import LongBench_samsum_datasets  # noqa: F401, F403
