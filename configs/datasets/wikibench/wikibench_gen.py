from mmengine.config import read_base

with read_base():
    from .wikibench_gen_f96ece import wikibench_datasets  # noqa: F401, F403
