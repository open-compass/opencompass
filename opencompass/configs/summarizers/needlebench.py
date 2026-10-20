
from opencompass.summarizers.needlebench import NeedleBenchSummarizer, NeedleBenchSummarizerV2


def create_m_rs_names_list(context_lengths, depths, needle_counts,
                           languages, dataset_size):
    names_dict = {}
    multi_needle_list = []
    multi_needle_en_list = []
    multi_needle_zh_list = []

    for needle_count in needle_counts:
        for language in languages:
            key = f'{needle_count}-Needle-{language.upper()}-{dataset_size.upper()}'
            names_list = [
                f'Length{length}Depth{int(depth)}_{needle_count}needle_{language}_{dataset_size}'
                for length in context_lengths
                for depth in depths
            ]
            names_dict[key] = names_list

            multi_needle_list.extend(names_list)
            if language == 'en':
                multi_needle_en_list.extend(names_list)
            elif language == 'zh':
                multi_needle_zh_list.extend(names_list)
    names_dict[f'Multi-Needle-Reasoning(M-RS)-{dataset_size.upper()}'] = multi_needle_list
    names_dict[f'Multi-Needle-Reasoning-EN-{dataset_size.upper()}'] = multi_needle_en_list
    names_dict[f'Multi-Needle-Reasoning-ZH-{dataset_size.upper()}'] = multi_needle_zh_list

    return names_dict

def create_summarizer(context_lengths, depths, dataset_size,
                      sparse_depths=None, mean=False):
    needle_counts = ['2', '3', '4', '5']
    languages = ['en', 'zh']
    if sparse_depths:
        depths = sparse_depths
    names_dict = {}
    multi_reasoning_names = create_m_rs_names_list(
        context_lengths, depths, needle_counts, languages, dataset_size)

    names_dict.update(multi_reasoning_names)

    single_needle_list = []
    single_needle_en_list = []
    single_needle_zh_list = []

    for language in languages:
        names_list = [
            [f'Length{length}_parallel_{language}_{dataset_size}', 'average_score']
            for length in context_lengths
            for depth in depths
        ]
        single_needle_list.extend(names_list)
        if language == 'en':
            single_needle_en_list.extend(names_list)
        elif language == 'zh':
            single_needle_zh_list.extend(names_list)
    names_dict[f'Single-Needle-Retrieval(S-RT)-{dataset_size.upper()}'] = single_needle_list
    names_dict[f'Single-Needle-Retrieval-EN-{dataset_size.upper()}'] = single_needle_en_list
    names_dict[f'Single-Needle-Retrieval-ZH-{dataset_size.upper()}'] = single_needle_zh_list

    parallel_list = []
    parallel_en_list = []
    parallel_zh_list = []

    for language in languages:
        names_list = [
            f'Length{length}_parallel_{language}_{dataset_size}'
            for length in context_lengths
        ]
        parallel_list.extend(names_list)
        if language == 'en':
            parallel_en_list.extend(names_list)
        elif language == 'zh':
            parallel_zh_list.extend(names_list)
    names_dict[f'Multi-Needle-Retrieval(M-RT)-{dataset_size.upper()}'] = parallel_list
    names_dict[f'Multi-Needle-Retrieval-EN-{dataset_size.upper()}'] = parallel_en_list
    names_dict[f'Multi-Needle-Retrieval-ZH-{dataset_size.upper()}'] = parallel_zh_list

    summary_groups = [
        {'name': key, 'subsets': value} for key, value in names_dict.items()
    ]
    if mean:
        summary_groups.append({
            'name': f'NeedleBench-Overall-Score-{dataset_size.upper()}',
            'subsets': [[f'Single-Needle-Retrieval(S-RT)-{dataset_size.upper()}', 'naive_average'],
                        [f'Multi-Needle-Reasoning(M-RS)-{dataset_size.upper()}', 'naive_average'],
                        [f'Multi-Needle-Retrieval(M-RT)-{dataset_size.upper()}', 'average_score']],
            'weights': {f'Single-Needle-Retrieval(S-RT)-{dataset_size.upper()}': 1/3,
                        f'Multi-Needle-Reasoning(M-RS)-{dataset_size.upper()}': 1/3,
                        f'Multi-Needle-Retrieval(M-RT)-{dataset_size.upper()}': 1/3}})
    else:
        summary_groups.append({
            'name': f'NeedleBench-Overall-Score-{dataset_size.upper()}',
            'subsets': [[f'Single-Needle-Retrieval(S-RT)-{dataset_size.upper()}', 'naive_average'],
                        [f'Multi-Needle-Reasoning(M-RS)-{dataset_size.upper()}', 'naive_average'],
                        [f'Multi-Needle-Retrieval(M-RT)-{dataset_size.upper()}', 'average_score']],
            'weights': {f'Single-Needle-Retrieval(S-RT)-{dataset_size.upper()}': 0.4,
                        f'Multi-Needle-Reasoning(M-RS)-{dataset_size.upper()}': 0.3,
                        f'Multi-Needle-Retrieval(M-RT)-{dataset_size.upper()}': 0.3}})
    summarizer_config = {
        'type': NeedleBenchSummarizerV2 if mean else NeedleBenchSummarizer,
        'summary_groups': summary_groups,
        'dataset_abbrs': [
            f'NeedleBench-Overall-Score-{dataset_size.upper()}',
            f'--------- NeedleBench-{dataset_size.upper()}-Single-Needle-Retrieval ---------',
            f'Single-Needle-Retrieval(S-RT)-{dataset_size.upper()}',
            f'Single-Needle-Retrieval-EN-{dataset_size.upper()}',
            f'Single-Needle-Retrieval-ZH-{dataset_size.upper()}',
            f'--------- NeedleBench-{dataset_size.upper()}-Multi-Needle-Retrieval ---------',
            f'Multi-Needle-Retrieval(M-RT)-{dataset_size.upper()}',
            f'Multi-Needle-Retrieval-EN-{dataset_size.upper()}',
            f'Multi-Needle-Retrieval-ZH-{dataset_size.upper()}',
            f'--------- NeedleBench-{dataset_size.upper()}-Multi-Needle-Reasoning ---------',
            f'Multi-Needle-Reasoning(M-RS)-{dataset_size.upper()}',
            f'Multi-Needle-Reasoning-EN-{dataset_size.upper()}',
            f'Multi-Needle-Reasoning-ZH-{dataset_size.upper()}',
            f'2-Needle-EN-{dataset_size.upper()}',
            f'2-Needle-ZH-{dataset_size.upper()}',
            f'3-Needle-EN-{dataset_size.upper()}',
            f'3-Needle-ZH-{dataset_size.upper()}',
            f'4-Needle-EN-{dataset_size.upper()}',
            f'4-Needle-ZH-{dataset_size.upper()}',
            f'5-Needle-EN-{dataset_size.upper()}',
            f'5-Needle-ZH-{dataset_size.upper()}',
            ]
        }
    return summarizer_config


depths = [0, 5, 10, 15, 21, 26, 31, 36, 42, 47, 52, 57, 63, 68, 73, 78, 84, 89, 94, 100]
depths_list_sparse = [0, 10, 21, 31, 42, 52, 63, 73, 84, 94, 100]

context_lengths_4k = list(range(1000, 5000, 1000))
needlebench_4k_summarizer = create_summarizer(context_lengths_4k, depths, '4k')
context_lengths_8k = list(range(5000, 9000, 1000))
needlebench_8k_summarizer = create_summarizer(context_lengths_8k, depths, '8k')
context_lengths_32k = [9000, 13000, 17000, 21000, 25000, 29000, 31000, 32000]
needlebench_32k_summarizer = create_summarizer(context_lengths_32k, depths_list_sparse, '32k')
context_lengths_128k = list([16000, 32000, 48000, 64000, 80000, 96000, 112000, 128000])
needlebench_128k_summarizer = create_summarizer(context_lengths_128k, depths_list_sparse, '128k')
context_lengths_200k = list([16000, 48000, 80000, 112000, 128000, 144000, 176000, 200000])
needlebench_200k_summarizer = create_summarizer(context_lengths_200k, depths_list_sparse, '200k')
context_lengths_256k = list([32000, 128000, 256000])
needlebench_256k_summarizer = create_summarizer(context_lengths_256k, depths_list_sparse, '256k')
context_lengths_1000k = list([20000, 160000, 300000, 440000, 580000, 720000, 860000, 1000000])
needlebench_1000k_summarizer = create_summarizer(context_lengths_1000k, depths_list_sparse, '1000k')

depths_list_internal = [0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100, ]
needlebench_internal_32k_summarizer = create_summarizer([32000], depths_list_internal, '32000')
needlebench_internal_100k_summarizer = create_summarizer([100000], depths_list_internal, '100000')
needlebench_internal_200k_summarizer = create_summarizer([200000], depths_list_internal, '200000')

depths_list_20 = [i for i in range(0, 101, 5)]  # [0, 5, 10, ..., 100]
depths_list_10 = [i for i in range(0, 101, 10)]  # [0, 10, 20, ..., 100]

context_lengths_4k = [1000, 2000, 3000, 4000]
needlebench_v2_4k_summarizer = create_summarizer(context_lengths_4k, depths_list_10, '4k', mean=True)
context_lengths_8k = [1000, 2000, 3000, 4000, 5000, 6000, 7000, 8000]
needlebench_v2_8k_summarizer = create_summarizer(context_lengths_8k, depths_list_10, '8k', mean=True)
context_lengths_32k = [1000, 4000, 8000, 12000, 16000, 20000, 24000, 28000, 32000]
needlebench_v2_32k_summarizer = create_summarizer(context_lengths_32k, depths_list_10, '32k', mean=True)
context_lengths_128k = [1000, 2000, 4000, 8000, 16000, 32000, 64000, 128000]
needlebench_v2_128k_summarizer = create_summarizer(context_lengths_128k, depths_list_10, '128k', mean=True)
context_lengths_200k = [16000, 48000, 80000, 112000, 128000, 144000, 176000, 200000]
needlebench_v2_200k_summarizer = create_summarizer(context_lengths_200k, depths_list_10, '200k', mean=True)
context_lengths_256k = [32000, 128000, 256000]
needlebench_v2_256k_summarizer = create_summarizer(context_lengths_256k, depths_list_10, '256k', mean=True)
context_lengths_1000k = [20000, 160000, 300000, 440000, 580000, 720000, 860000, 1000000]
needlebench_v2_1000k_summarizer = create_summarizer(context_lengths_1000k, depths_list_10, '1000k', mean=True)