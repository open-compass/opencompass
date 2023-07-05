from mmengine.config import read_base

with read_base():
    from .piqa_ppl_788dbe import piqa_datasets  # noqa: F401, F403
