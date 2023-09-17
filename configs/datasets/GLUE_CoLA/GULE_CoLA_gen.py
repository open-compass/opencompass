from mmengine.config import read_base

with read_base():
    from .GULE_CoLA_gen_c68cef import CoLA_datasets  # noqa: F401, F403
