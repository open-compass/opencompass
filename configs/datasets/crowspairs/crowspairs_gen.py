from mmengine.config import read_base

with read_base():
    from .crowspairs_gen_dd110a import crowspairs_datasets  # noqa: F401, F403
