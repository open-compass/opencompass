from mmengine.config import read_base

with read_base():
    from .hle_llmjudge_gen_63a000 import hle_datasets  # noqa: F401, F403
