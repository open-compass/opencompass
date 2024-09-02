from mmengine.config import read_base

with read_base():
    from .humanevalx_gen_620cfa import humanevalx_datasets  # noqa: F401, F403
