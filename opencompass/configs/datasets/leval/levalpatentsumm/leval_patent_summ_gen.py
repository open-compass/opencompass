from mmengine.config import read_base

with read_base():
    from .leval_patent_summ_gen_b03798 import LEval_patent_summ_datasets  # noqa: F401, F403
