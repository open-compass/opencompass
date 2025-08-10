from mmengine.config import read_base

with read_base():
    from .infinitebench_enmc_gen_3a4102 import InfiniteBench_enmc_datasets  # noqa: F401, F403
