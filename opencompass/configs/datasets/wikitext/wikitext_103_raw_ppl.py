from mmengine.config import read_base

with read_base():
    from .wikitext_103_raw_ppl_05247d import wikitext_103_raw_datasets  # noqa: F401, F403
