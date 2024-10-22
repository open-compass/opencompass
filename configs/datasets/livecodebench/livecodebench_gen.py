from mmengine.config import read_base

with read_base():
    from .livecodebench_gen_b2b0fd import LCB_datasets  # noqa: F401, F403
