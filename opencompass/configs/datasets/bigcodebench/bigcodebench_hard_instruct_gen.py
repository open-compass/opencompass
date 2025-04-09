from mmengine.config import read_base

with read_base():
    from .bigcodebench_hard_instruct_gen_c3d5ad import bigcodebench_hard_instruct_datasets  # noqa: F401, F403
