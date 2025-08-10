from mmengine.config import read_base

with read_base():
    from .FinanceIQ_ppl_42b9bd import financeIQ_datasets  # noqa: F401, F403
