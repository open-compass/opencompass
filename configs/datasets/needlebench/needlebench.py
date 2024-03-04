from mmengine.config import read_base

with read_base():
    from .needlebench_4k.needlebench import needlebench_datasets as needlebench_datasets_4k
    from .needlebench_8k.needlebench import needlebench_datasets as needlebench_datasets_8k
    from .needlebench_32k.needlebench import needlebench_datasets as needlebench_datasets_32k
    from .needlebench_128k.needlebench import needlebench_datasets as needlebench_datasets_128k
    from .needlebench_200k.needlebench import needlebench_datasets as needlebench_datasets_200k

needlebench_datasets = sum((v for k, v in locals().items() if k.endswith('_datasets')), [])
