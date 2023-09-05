from mmengine.config import read_base

with read_base():
    from .longbench_nq_gen_d30cb9 import LongBench_nq_datasets  # noqa: F401, F403
