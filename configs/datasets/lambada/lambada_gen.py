from mmengine.config import read_base

with read_base():
    from .lambada_gen_7ffe3d import lambada_datasets  # noqa: F401, F403
