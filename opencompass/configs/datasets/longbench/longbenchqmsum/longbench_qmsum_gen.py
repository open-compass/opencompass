from mmengine.config import read_base

with read_base():
    from .longbench_qmsum_gen_d33331 import LongBench_qmsum_datasets  # noqa: F401, F403
