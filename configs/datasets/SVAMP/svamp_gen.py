from mmengine.config import read_base

with read_base():
    from .svamp_gen_5edbdb import svamp_datasets  # noqa: F401, F403
