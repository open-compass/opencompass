from mmengine.config import read_base

with read_base():
    from .bigcodebench_full_instruct_gen_8815eb import bigcodebench_full_instruct_datasets  # noqa: F401, F403
