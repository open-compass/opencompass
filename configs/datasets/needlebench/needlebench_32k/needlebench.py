from mmengine.config import read_base

with read_base():
    from .needlebench_multi_reasoning import needlebench_datasets_2needle_en as needlebench_multi_2needle_en_datasets
    from .needlebench_multi_reasoning import needlebench_datasets_3needle_en as needlebench_multi_3needle_en_datasets
    from .needlebench_multi_reasoning import needlebench_datasets_4needle_en as needlebench_multi_4needle_en_datasets
    from .needlebench_multi_reasoning import needlebench_datasets_5needle_en as needlebench_multi_5needle_en_datasets
    from .needlebench_multi_reasoning import needlebench_datasets_2needle_zh as needlebench_multi_2needle_zh_datasets
    from .needlebench_multi_reasoning import needlebench_datasets_3needle_zh as needlebench_multi_3needle_zh_datasets
    from .needlebench_multi_reasoning import needlebench_datasets_4needle_zh as needlebench_multi_4needle_zh_datasets
    from .needlebench_multi_reasoning import needlebench_datasets_5needle_zh as needlebench_multi_5needle_zh_datasets

    from .needlebench_single import needlebench_datasets_en as needlebench_origin_en_datasets
    from .needlebench_single import needlebench_datasets_zh as needlebench_origin_zh_datasets
    from .needlebench_multi_retrieval import needlebench_datasets_en as needlebench_parallel_en_datasets
    from .needlebench_multi_retrieval import needlebench_datasets_zh as needlebench_parallel_zh_datasets

needlebench_datasets = sum((v for k, v in locals().items() if k.endswith('_datasets')), [])
