from mmengine.config import read_base

with read_base():
    from .FewCLUE_bustm_ppl_47f2ab import bustm_datasets  # noqa: F401, F403
