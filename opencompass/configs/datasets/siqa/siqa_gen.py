from mmengine.config import read_base

with read_base():
    from .siqa_gen_18632c import siqa_datasets  # noqa: F401, F403
