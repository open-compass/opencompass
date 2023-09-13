from mmengine.config import read_base

with read_base():
    from .scibench_gen_5k76h2 import scibench_datasets  # noqa: F401, F403
