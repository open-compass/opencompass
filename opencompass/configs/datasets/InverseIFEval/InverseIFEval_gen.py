from mmengine.config import read_base

with read_base():
    from .InverseIFEval_rawprompt_gen import inverse_ifeval_datasets  # noqa: F401, F403
