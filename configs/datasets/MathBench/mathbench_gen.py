from mmengine.config import read_base

with read_base():
    from .mathbench_gen_7b734b import mathbench_datasets  # noqa: F401, F403
