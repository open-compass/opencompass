from mmengine.config import read_base

with read_base():
    from .longbench_multi_news_gen_f6e3fb import LongBench_multi_news_datasets  # noqa: F401, F403
