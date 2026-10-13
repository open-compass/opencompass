from mmengine.config import read_base

with read_base():
    from .bigcodebench_hard_instruct_gen import bigcodebench_hard_instruct_datasets
    from .bigcodebench_hard_complete_gen import bigcodebench_hard_complete_datasets

bigcodebench_hard_datasets = sum((v for k, v in locals().items() if k.endswith('_datasets')), [])
