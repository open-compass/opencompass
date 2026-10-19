from mmengine.config import read_base

with read_base():
    from .medbench_gen_0b4fff import medbench_datasets  # noqa: F401, F403
