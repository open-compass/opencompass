from mmengine.config import read_base

with read_base():
    from opencompass.configs.datasets.needlebench_v2.needlebench_v2_128k.needlebench_v2_multi_reasoning_128k import needlebench_2needle_en_datasets as needlebench_multi_2needle_en_datasets
    from opencompass.configs.datasets.needlebench_v2.needlebench_v2_128k.needlebench_v2_multi_reasoning_128k import needlebench_3needle_en_datasets as needlebench_multi_3needle_en_datasets
    from opencompass.configs.datasets.needlebench_v2.needlebench_v2_128k.needlebench_v2_multi_reasoning_128k import needlebench_4needle_en_datasets as needlebench_multi_4needle_en_datasets
    from opencompass.configs.datasets.needlebench_v2.needlebench_v2_128k.needlebench_v2_multi_reasoning_128k import needlebench_5needle_en_datasets as needlebench_multi_5needle_en_datasets
    from opencompass.configs.datasets.needlebench_v2.needlebench_v2_128k.needlebench_v2_multi_reasoning_128k import needlebench_2needle_zh_datasets as needlebench_multi_2needle_zh_datasets
    from opencompass.configs.datasets.needlebench_v2.needlebench_v2_128k.needlebench_v2_multi_reasoning_128k import needlebench_3needle_zh_datasets as needlebench_multi_3needle_zh_datasets
    from opencompass.configs.datasets.needlebench_v2.needlebench_v2_128k.needlebench_v2_multi_reasoning_128k import needlebench_4needle_zh_datasets as needlebench_multi_4needle_zh_datasets
    from opencompass.configs.datasets.needlebench_v2.needlebench_v2_128k.needlebench_v2_multi_reasoning_128k import needlebench_5needle_zh_datasets as needlebench_multi_5needle_zh_datasets

    from opencompass.configs.datasets.needlebench_v2.needlebench_v2_128k.needlebench_v2_single_128k import needlebench_en_datasets as needlebench_origin_en_datasets
    from opencompass.configs.datasets.needlebench_v2.needlebench_v2_128k.needlebench_v2_single_128k import needlebench_zh_datasets as needlebench_origin_zh_datasets
    from opencompass.configs.datasets.needlebench_v2.needlebench_v2_128k.needlebench_v2_multi_retrieval_128k import needlebench_en_datasets as needlebench_parallel_en_datasets
    from opencompass.configs.datasets.needlebench_v2.needlebench_v2_128k.needlebench_v2_multi_retrieval_128k import needlebench_zh_datasets as needlebench_parallel_zh_datasets

needlebench_datasets = sum((v for k, v in locals().items() if k.endswith('_datasets')), [])


if __name__ == '__main__':
    print(len(needlebench_datasets))
    # sum num_repeats_per_file of all datasets
    num_repeats_per_file = sum(dataset['num_repeats_per_file'] for dataset in needlebench_datasets) * 8
    print(num_repeats_per_file)
    # every repeat is 5 seconds
    print(num_repeats_per_file * 5 / 60, 'minutes')
    # print number of hours
    print(num_repeats_per_file * 5 / 3600, 'hours')

    # if every repeat is 2 minutes, how many days
    print(num_repeats_per_file * 2 / 60 / 24, 'days')
