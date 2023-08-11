from mmengine.config import read_base

with read_base():
    from .LEval_tvshow_summ_gen_rouge import LEval_tvshow_summ_datasets  # noqa: F401, F403
