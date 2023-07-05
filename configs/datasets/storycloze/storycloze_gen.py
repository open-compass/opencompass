from mmengine.config import read_base

with read_base():
    from .storycloze_gen_c5a230 import storycloze_datasets  # noqa: F401, F403
