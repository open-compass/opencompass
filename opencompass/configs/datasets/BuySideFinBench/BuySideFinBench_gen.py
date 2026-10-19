from mmengine.config import read_base

with read_base():
    from .BuySideFinBench_gen_ca1704 import BuySideFinBench_datasets  # noqa: F401, F403
