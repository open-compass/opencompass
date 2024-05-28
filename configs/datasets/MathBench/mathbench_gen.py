from mmengine.config import read_base

with read_base():
    from .mathbench_2024_gen_1dc21d import mathbench_datasets  # noqa: F401, F403
