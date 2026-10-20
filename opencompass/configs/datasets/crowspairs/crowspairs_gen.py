from mmengine.config import read_base

with read_base():
    from .crowspairs_gen_381af0 import crowspairs_datasets  # noqa: F401, F403
