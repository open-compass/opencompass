from mmengine.config import read_base

with read_base():
    from .longbench_lcc_gen_6ba507 import LongBench_lcc_datasets  # noqa: F401, F403
