from mmengine.config import read_base

with read_base():
    from .civilcomments_clp_a3c5fd import civilcomments_datasets  # noqa: F401, F403
