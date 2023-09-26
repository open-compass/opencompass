from mmengine.config import read_base

with read_base():
    from .GULE_CoLA_ppl_f165d7 import CoLA_datasets  # noqa: F401, F403
