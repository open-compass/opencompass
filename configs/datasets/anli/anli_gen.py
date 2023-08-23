from mmengine.config import read_base

with read_base():
    from .anli_gen_fc7328 import anli_datasets  # noqa: F401, F403
