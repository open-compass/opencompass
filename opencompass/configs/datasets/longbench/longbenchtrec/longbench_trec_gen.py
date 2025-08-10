from mmengine.config import read_base

with read_base():
    from .longbench_trec_gen_824187 import LongBench_trec_datasets  # noqa: F401, F403
