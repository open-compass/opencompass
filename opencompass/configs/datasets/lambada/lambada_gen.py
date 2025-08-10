from mmengine.config import read_base

with read_base():
    from .lambada_gen_217e11 import lambada_datasets  # noqa: F401, F403
