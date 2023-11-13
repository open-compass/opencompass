from mmengine.config import read_base

with read_base():
    from .mathbench_gen_10da90 import mathbench_datasets  # noqa: F401, F403
