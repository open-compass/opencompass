from mmengine.config import read_base

with read_base():
    from .GLUE_CoLA_ppl_77d0df import CoLA_datasets  # noqa: F401, F403
