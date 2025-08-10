from mmengine.config import read_base

with read_base():

    from .needlebench_single import needlebench_en_datasets as needlebench_origin_en_datasets
    from .needlebench_single import needlebench_zh_datasets as needlebench_origin_zh_datasets

needlebench_datasets = sum((v for k, v in locals().items() if k.endswith('_datasets')), [])
