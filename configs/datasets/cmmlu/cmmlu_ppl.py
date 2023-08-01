from mmengine.config import read_base

with read_base():
    from .cmmlu_ppl_fd1f2f import cmmlu_datasets  # noqa: F401, F403
