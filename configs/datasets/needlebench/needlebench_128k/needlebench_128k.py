from mmengine.config import read_base

with read_base():
    from .needlebench_multi_reasoning_128k import needlebench_2needle_en_datasets as needlebench_multi_2needle_en_datasets
    from .needlebench_multi_reasoning_128k import needlebench_3needle_en_datasets as needlebench_multi_3needle_en_datasets
    from .needlebench_multi_reasoning_128k import needlebench_4needle_en_datasets as needlebench_multi_4needle_en_datasets
    from .needlebench_multi_reasoning_128k import needlebench_5needle_en_datasets as needlebench_multi_5needle_en_datasets
    from .needlebench_multi_reasoning_128k import needlebench_2needle_zh_datasets as needlebench_multi_2needle_zh_datasets
    from .needlebench_multi_reasoning_128k import needlebench_3needle_zh_datasets as needlebench_multi_3needle_zh_datasets
    from .needlebench_multi_reasoning_128k import needlebench_4needle_zh_datasets as needlebench_multi_4needle_zh_datasets
    from .needlebench_multi_reasoning_128k import needlebench_5needle_zh_datasets as needlebench_multi_5needle_zh_datasets

    from .needlebench_single_128k import needlebench_en_datasets as needlebench_origin_en_datasets
    from .needlebench_single_128k import needlebench_zh_datasets as needlebench_origin_zh_datasets
    from .needlebench_multi_retrieval_128k import needlebench_en_datasets as needlebench_parallel_en_datasets
    from .needlebench_multi_retrieval_128k import needlebench_zh_datasets as needlebench_parallel_zh_datasets

needlebench_datasets = sum((v for k, v in locals().items() if k.endswith('_datasets')), [])
