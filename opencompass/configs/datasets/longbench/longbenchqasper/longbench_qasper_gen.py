from mmengine.config import read_base

with read_base():
    from .longbench_qasper_gen_6b3efc import LongBench_qasper_datasets  # noqa: F401, F403
