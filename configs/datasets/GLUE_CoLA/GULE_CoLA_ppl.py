from mmengine.config import read_base

with read_base():
    from .GULE_CoLA_ppl_77d0df import CoLA_datasets  # noqa: F401, F403
