from mmengine.config import read_base

with read_base():
    from .longbench_passage_retrieval_zh_gen_01cca2 import LongBench_passage_retrieval_zh_datasets  # noqa: F401, F403
