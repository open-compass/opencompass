from mmengine.config import read_base

with read_base():
    from .longbench_passage_retrieval_en_gen_734db5 import LongBench_passage_retrieval_en_datasets  # noqa: F401, F403
