from mmengine.config import read_base

with read_base():
    from .matbench_gen_f71840 import matbench_datasets  # noqa: F401, F403
