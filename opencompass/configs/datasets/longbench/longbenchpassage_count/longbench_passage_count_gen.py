from mmengine.config import read_base

with read_base():
    from .longbench_passage_count_gen_dcdaab import LongBench_passage_count_datasets  # noqa: F401, F403
