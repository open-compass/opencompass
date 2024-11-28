from mmengine.config import read_base

with read_base():
    from .cmo_fib_gen_ace24b import cmo_fib_datasets  # noqa: F401, F403