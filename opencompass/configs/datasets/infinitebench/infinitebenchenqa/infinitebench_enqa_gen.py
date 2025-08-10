from mmengine.config import read_base

with read_base():
    from .infinitebench_enqa_gen_a1640c import InfiniteBench_enqa_datasets  # noqa: F401, F403
