from mmengine.config import read_base

with read_base():
    from .py150_gen_38b13d import py150_datasets  # noqa: F401, F403
