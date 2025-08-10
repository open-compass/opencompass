from mmengine.config import read_base

with read_base():
    from .infinitebench_zhqa_gen_1e5293 import InfiniteBench_zhqa_datasets  # noqa: F401, F403
