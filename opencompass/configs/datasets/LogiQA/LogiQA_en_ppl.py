from mmengine.config import read_base

with read_base():
    from .LogiQA_en_ppl_20dfb3 import LogiQA_en_datasets  # noqa: F401, F403
