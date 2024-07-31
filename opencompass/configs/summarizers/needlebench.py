from opencompass.summarizers.needlebench import NeedleBenchSummarizer


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
    names_dict['Multi-Needle-Reasoning(M-RS)'] =  multi_needle_list
    names_dict['Multi-Needle-Reasoning-EN'] = multi_needle_en_list
    names_dict['Multi-Needle-Reasoning-ZH'] = multi_needle_zh_list

    return names_dict

def create_summarizer(context_lengths, depths, dataset_size,
                      sparse_depths=None):
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
            f'Length{length}Depth{int(depth)}_origin_{language}_{dataset_size}'
            for length in context_lengths
            for depth in depths
        ]
        single_needle_list.extend(names_list)
        if language == 'en':
            single_needle_en_list.extend(names_list)
        elif language == 'zh':
            single_needle_zh_list.extend(names_list)
    names_dict['Single-Needle-Retrieval(S-RT)'] = single_needle_list
    names_dict['Single-Needle-Retrieval-EN'] = single_needle_en_list
    names_dict['Single-Needle-Retrieval-ZH'] = single_needle_zh_list

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
    names_dict['Multi-Needle-Retrieval(M-RT)'] = parallel_list
    names_dict['Multi-Needle-Retrieval-EN'] = parallel_en_list
    names_dict['Multi-Needle-Retrieval-ZH'] = parallel_zh_list

    summary_groups = [
        {'name': key, 'subsets': value} for key, value in names_dict.items()
    ]

    summary_groups.append({
        'name': 'NeedleBench-Overall-Score',
        'subsets': [['Single-Needle-Retrieval(S-RT)', 'naive_average'],
                    ['Multi-Needle-Reasoning(M-RS)', 'naive_average'],
                    ['Multi-Needle-Retrieval(M-RT)', 'average_score']],
        'weights': {'Single-Needle-Retrieval(S-RT)': 0.4,
                    'Multi-Needle-Reasoning(M-RS)': 0.3,
                    'Multi-Needle-Retrieval(M-RT)': 0.3}})
    summarizer_config = {
        'type': NeedleBenchSummarizer,
        'summary_groups': summary_groups,
        'dataset_abbrs': [
            'NeedleBench-Overall-Score',
            f'--------- NeedleBench-{dataset_size.upper()}-Single-Needle-Retrieval ---------',
            'Single-Needle-Retrieval(S-RT)',
            'Single-Needle-Retrieval-EN',
            'Single-Needle-Retrieval-ZH',
            f'--------- NeedleBench-{dataset_size.upper()}-Multi-Needle-Retrieval ---------',
            'Multi-Needle-Retrieval(M-RT)',
            'Multi-Needle-Retrieval-EN',
            'Multi-Needle-Retrieval-ZH',
            f'--------- NeedleBench-{dataset_size.upper()}-Multi-Needle-Reasoning ---------',
            'Multi-Needle-Reasoning(M-RS)',
            'Multi-Needle-Reasoning-EN',
            'Multi-Needle-Reasoning-ZH',
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


_needlebench_8k_parallel_en_batch1 = []
_needlebench_8k_parallel_en_batch5 = []
_needlebench_8k_parallel_en_batch10 = []
_needlebench_8k_parallel_en_batch15 = []
_needlebench_8k_parallel_en_batch20 = []
_needlebench_8k_parallel_zh_batch1 = []
_needlebench_8k_parallel_zh_batch5 = []
_needlebench_8k_parallel_zh_batch10 = []
_needlebench_8k_parallel_zh_batch15 = []
_needlebench_8k_parallel_zh_batch20 = []
for original_context_length in context_lengths_8k:
    _needlebench_8k_parallel_en_batch1.append(f'Length{original_context_length}_parallel_en_8k_batch1')
    _needlebench_8k_parallel_en_batch5.append(f'Length{original_context_length}_parallel_en_8k_batch5')
    _needlebench_8k_parallel_en_batch10.append(f'Length{original_context_length}_parallel_en_8k_batch10')
    _needlebench_8k_parallel_en_batch15.append(f'Length{original_context_length}_parallel_en_8k_batch15')
    _needlebench_8k_parallel_en_batch20.append(f'Length{original_context_length}_parallel_en_8k_batch20')
    _needlebench_8k_parallel_zh_batch1.append(f'Length{original_context_length}_parallel_zh_8k_batch1')
    _needlebench_8k_parallel_zh_batch5.append(f'Length{original_context_length}_parallel_zh_8k_batch5')
    _needlebench_8k_parallel_zh_batch10.append(f'Length{original_context_length}_parallel_zh_8k_batch10')
    _needlebench_8k_parallel_zh_batch15.append(f'Length{original_context_length}_parallel_zh_8k_batch15')
    _needlebench_8k_parallel_zh_batch20.append(f'Length{original_context_length}_parallel_zh_8k_batch20')


_needlebench_8k_parallel_batch1 = _needlebench_8k_parallel_en_batch1 + _needlebench_8k_parallel_zh_batch1
_needlebench_8k_parallel_batch5 = _needlebench_8k_parallel_en_batch5 + _needlebench_8k_parallel_zh_batch5
_needlebench_8k_parallel_batch10 = _needlebench_8k_parallel_en_batch10 + _needlebench_8k_parallel_zh_batch10
_needlebench_8k_parallel_batch15 = _needlebench_8k_parallel_en_batch15 + _needlebench_8k_parallel_zh_batch15
_needlebench_8k_parallel_batch20 = _needlebench_8k_parallel_en_batch20 + _needlebench_8k_parallel_zh_batch20

needlebench_summary_groups = [
    {'name': 'parallel_version_batch1', 'subsets': [[_dataset, 'average_score'] for _dataset in _needlebench_8k_parallel_batch1]},
    {'name': 'parallel_version_zh_batch1', 'subsets': [[_dataset, 'average_score'] for _dataset in _needlebench_8k_parallel_zh_batch1]},
    {'name': 'parallel_version_en_batch1', 'subsets': [[_dataset, 'average_score'] for _dataset in _needlebench_8k_parallel_en_batch1]},
    {'name': 'parallel_version_batch5', 'subsets': [[_dataset, 'average_score'] for _dataset in _needlebench_8k_parallel_batch5]},
    {'name': 'parallel_version_zh_batch5', 'subsets': [[_dataset, 'average_score'] for _dataset in _needlebench_8k_parallel_zh_batch5]},
    {'name': 'parallel_version_en_batch5', 'subsets': [[_dataset, 'average_score'] for _dataset in _needlebench_8k_parallel_en_batch5]},
    {'name': 'parallel_version_batch10', 'subsets': [[_dataset, 'average_score'] for _dataset in _needlebench_8k_parallel_batch10]},
    {'name': 'parallel_version_zh_batch10', 'subsets': [[_dataset, 'average_score'] for _dataset in _needlebench_8k_parallel_zh_batch10]},
    {'name': 'parallel_version_en_batch10', 'subsets': [[_dataset, 'average_score'] for _dataset in _needlebench_8k_parallel_en_batch10]},
    {'name': 'parallel_version_batch15', 'subsets': [[_dataset, 'average_score'] for _dataset in _needlebench_8k_parallel_batch15]},
    {'name': 'parallel_version_zh_batch15', 'subsets': [[_dataset, 'average_score'] for _dataset in _needlebench_8k_parallel_zh_batch15]},
    {'name': 'parallel_version_en_batch15', 'subsets': [[_dataset, 'average_score'] for _dataset in _needlebench_8k_parallel_en_batch15]},
    {'name': 'parallel_version_batch20', 'subsets': [[_dataset, 'average_score'] for _dataset in _needlebench_8k_parallel_batch20]},
    {'name': 'parallel_version_zh_batch20', 'subsets': [[_dataset, 'average_score'] for _dataset in _needlebench_8k_parallel_zh_batch20]},
    {'name': 'parallel_version_en_batch20', 'subsets': [[_dataset, 'average_score'] for _dataset in _needlebench_8k_parallel_en_batch20]},
]

needlebench_8k_batch_overall_summarizer = dict(
    dataset_abbrs=[
        '--------- NeedleBench-8k Parallel-Needles ---------',  # category
        'parallel_version_batch1',
        'parallel_version_batch5',
        'parallel_version_batch10',
        'parallel_version_batch15',
        'parallel_version_batch20',
        'parallel_version_zh_batch1',
        'parallel_version_en_batch1',
        'parallel_version_zh_batch5',
        'parallel_version_en_batch5',
        'parallel_version_zh_batch10',
        'parallel_version_en_batch10',
        'parallel_version_zh_batch15',
        'parallel_version_en_batch15',
        'parallel_version_zh_batch20',
        'parallel_version_en_batch20',
    ],
    summary_groups=needlebench_summary_groups,
)

needlebench_summary_groups = [
    {'name': 'parallel_version_batch1', 'subsets': [[_dataset, 'Depth0'] for _dataset in _needlebench_8k_parallel_batch1]},
    {'name': 'parallel_version_zh_batch1', 'subsets': [[_dataset, 'Depth0'] for _dataset in _needlebench_8k_parallel_zh_batch1]},
    {'name': 'parallel_version_en_batch1', 'subsets': [[_dataset, 'Depth0'] for _dataset in _needlebench_8k_parallel_en_batch1]},
    {'name': 'parallel_version_batch5', 'subsets': [[_dataset, 'Depth0'] for _dataset in _needlebench_8k_parallel_batch5]},
    {'name': 'parallel_version_zh_batch5', 'subsets': [[_dataset, 'Depth0'] for _dataset in _needlebench_8k_parallel_zh_batch5]},
    {'name': 'parallel_version_en_batch5', 'subsets': [[_dataset, 'Depth0'] for _dataset in _needlebench_8k_parallel_en_batch5]},
    {'name': 'parallel_version_batch10', 'subsets': [[_dataset, 'Depth0'] for _dataset in _needlebench_8k_parallel_batch10]},
    {'name': 'parallel_version_zh_batch10', 'subsets': [[_dataset, 'Depth0'] for _dataset in _needlebench_8k_parallel_zh_batch10]},
    {'name': 'parallel_version_en_batch10', 'subsets': [[_dataset, 'Depth0'] for _dataset in _needlebench_8k_parallel_en_batch10]},
    {'name': 'parallel_version_batch15', 'subsets': [[_dataset, 'Depth0'] for _dataset in _needlebench_8k_parallel_batch15]},
    {'name': 'parallel_version_zh_batch15', 'subsets': [[_dataset, 'Depth0'] for _dataset in _needlebench_8k_parallel_zh_batch15]},
    {'name': 'parallel_version_en_batch15', 'subsets': [[_dataset, 'Depth0'] for _dataset in _needlebench_8k_parallel_en_batch15]},
    {'name': 'parallel_version_batch20', 'subsets': [[_dataset, 'Depth0'] for _dataset in _needlebench_8k_parallel_batch20]},
    {'name': 'parallel_version_zh_batch20', 'subsets': [[_dataset, 'Depth0'] for _dataset in _needlebench_8k_parallel_zh_batch20]},
    {'name': 'parallel_version_en_batch20', 'subsets': [[_dataset, 'Depth0'] for _dataset in _needlebench_8k_parallel_en_batch20]},
]

needlebench_8k_batch_depth0_summarizer = dict(
    dataset_abbrs=[
        '--------- NeedleBench-8k Parallel-Needles ---------',  # category
        'parallel_version_batch1',
        'parallel_version_batch5',
        'parallel_version_batch10',
        'parallel_version_batch15',
        'parallel_version_batch20',
        'parallel_version_zh_batch1',
        'parallel_version_en_batch1',
        'parallel_version_zh_batch5',
        'parallel_version_en_batch5',
        'parallel_version_zh_batch10',
        'parallel_version_en_batch10',
        'parallel_version_zh_batch15',
        'parallel_version_en_batch15',
        'parallel_version_zh_batch20',
        'parallel_version_en_batch20',
    ],
    summary_groups=needlebench_summary_groups,
)

def gen_atc_summarizer(needle_num_list):
    categories = [
        'ZH-Direct-CE', 'EN-Direct-CE',
        'ZH-Reasoning-CE', 'EN-Reasoning-CE'
    ]
    needlebench_atc_summary_groups = []

    # 根据分类生成summary groups
    for category in categories:
        # 对于CircularEval相关的评分，使用perf_4指标，否则使用acc_1指标
        metric = 'perf_4' if 'CE' in category else 'acc_1'
        # 生成subsets时，不需要在数据集名称中包含CircularEval信息
        cleaned_category = category.replace('-CE', '').replace('-Direct', '')
        needlebench_atc_summary_groups.append({
            'name': category,
            'subsets': [
                [f'NeedleBenchATCDataset-{num_needles}Needle-{cleaned_category}', metric]
                for num_needles in needle_num_list
            ],
            'weights': {f'NeedleBenchATCDataset-{num_needles}Needle-{cleaned_category}': num_needles for num_needles in needle_num_list},
        })

    needlebench_atc_summary_groups.append({
        'name': 'ATC-CE-Overall',
        'subsets': [
            [f'{category}', 'weighted_average'] for category in categories
        ],
    })
    atc_dataset_abbrs = []
    atc_dataset_abbrs.append(['ATC-CE-Overall', 'naive_average'])

    for category in categories:
        weighted_average_score_entry = [f'{category}', 'weighted_average']
        atc_dataset_abbrs.append(weighted_average_score_entry)

    needlebench_atc_summarizer = dict(
        dataset_abbrs=[
            *atc_dataset_abbrs,
            '######## Needlebench-ATC Accuracy ########',  # category
            *[[f'NeedleBenchATCDataset-{num_needles}Needle-ZH', 'acc_1'] for num_needles in needle_num_list],
            '------------------------------------------',
            *[[f'NeedleBenchATCDataset-{num_needles}Needle-EN', 'acc_1'] for num_needles in needle_num_list],
            '------------------------------------------',
            *[[f'NeedleBenchATCDataset-{num_needles}Needle-ZH-Reasoning', 'acc_1'] for num_needles in needle_num_list],
            '------------------------------------------',
            *[[f'NeedleBenchATCDataset-{num_needles}Needle-EN-Reasoning', 'acc_1'] for num_needles in needle_num_list],
            '------------------------------------------',
            '######## Needlebench-ATC CircularEval ########',  # category
            *[[f'NeedleBenchATCDataset-{num_needles}Needle-ZH', 'perf_4'] for num_needles in needle_num_list],
            '------------------------------------------',
            *[[f'NeedleBenchATCDataset-{num_needles}Needle-EN', 'perf_4'] for num_needles in needle_num_list],
            '------------------------------------------',
            *[[f'NeedleBenchATCDataset-{num_needles}Needle-ZH-Reasoning', 'perf_4'] for num_needles in needle_num_list],
            '------------------------------------------',
            *[[f'NeedleBenchATCDataset-{num_needles}Needle-EN-Reasoning', 'perf_4'] for num_needles in needle_num_list],
            '------------------------------------------',
        ],
        summary_groups=needlebench_atc_summary_groups
    )
    return needlebench_atc_summarizer


atc_summarizer_20 = gen_atc_summarizer(list(range(2, 20, 1)))
atc_summarizer_50 = gen_atc_summarizer(list(range(2, 50, 1)))
atc_summarizer_80 = gen_atc_summarizer(list(range(2, 80, 1)))
