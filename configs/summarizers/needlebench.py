from opencompass.summarizers.needlebench import NeedleBenchSummarizer
from opencompass.summarizers.needlebench import NeedleBenchATCSummarizer

# ----------NeedleBench-4k-summarizer----------
context_lengths_4k = list(range(1000, 5000, 1000))
depths = [0, 5, 10, 15, 21, 26, 31, 36, 42, 47, 52, 57, 63, 68, 73, 78, 84, 89, 94, 100]
depths_list_sparse = [0, 10, 21, 31, 42, 52, 63, 73, 84, 94, 100]

# Initialize the lists
_needlebench_4k_2needle_en = []
_needlebench_4k_3needle_en = []
_needlebench_4k_4needle_en = []
_needlebench_4k_5needle_en = []
_needlebench_4k_2needle_zh = []
_needlebench_4k_3needle_zh = []
_needlebench_4k_4needle_zh = []
_needlebench_4k_5needle_zh = []
_needlebench_4k_origin_en = []
_needlebench_4k_origin_zh = []

# Fill the lists using nested loops
for original_context_length in context_lengths_4k:
    for depth_percent in depths:
        _needlebench_4k_2needle_en.append(f'Length{original_context_length}Depth{int(depth_percent)}_2needle_en_4k')
        _needlebench_4k_3needle_en.append(f'Length{original_context_length}Depth{int(depth_percent)}_3needle_en_4k')
        _needlebench_4k_4needle_en.append(f'Length{original_context_length}Depth{int(depth_percent)}_4needle_en_4k')
        _needlebench_4k_5needle_en.append(f'Length{original_context_length}Depth{int(depth_percent)}_5needle_en_4k')
        _needlebench_4k_2needle_zh.append(f'Length{original_context_length}Depth{int(depth_percent)}_2needle_zh_4k')
        _needlebench_4k_3needle_zh.append(f'Length{original_context_length}Depth{int(depth_percent)}_3needle_zh_4k')
        _needlebench_4k_4needle_zh.append(f'Length{original_context_length}Depth{int(depth_percent)}_4needle_zh_4k')
        _needlebench_4k_5needle_zh.append(f'Length{original_context_length}Depth{int(depth_percent)}_5needle_zh_4k')

        _needlebench_4k_origin_en.append(f'Length{original_context_length}Depth{int(depth_percent)}_origin_en_4k')
        _needlebench_4k_origin_zh.append(f'Length{original_context_length}Depth{int(depth_percent)}_origin_zh_4k')

# Concatenate the multi-needle and origin lists
_needlebench_4k_multi_needle_en = _needlebench_4k_2needle_en + _needlebench_4k_3needle_en + _needlebench_4k_4needle_en + _needlebench_4k_5needle_en
_needlebench_4k_multi_needle_zh = _needlebench_4k_2needle_zh + _needlebench_4k_3needle_zh + _needlebench_4k_4needle_zh + _needlebench_4k_5needle_zh
_needlebench_4k_origin = _needlebench_4k_origin_en + _needlebench_4k_origin_zh
_needlebench_4k_multi_needle = _needlebench_4k_multi_needle_en + _needlebench_4k_multi_needle_zh

# Repeating the same process for parallel (assuming it's similar to origin_en)
_needlebench_4k_parallel_en = []
_needlebench_4k_parallel_zh = []
for original_context_length in context_lengths_4k:
    _needlebench_4k_parallel_en.append(f'Length{original_context_length}_parallel_en_4k')
for original_context_length in context_lengths_4k:
    _needlebench_4k_parallel_zh.append(f'Length{original_context_length}_parallel_zh_4k')
_needlebench_4k_parallel = _needlebench_4k_parallel_en + _needlebench_4k_parallel_zh

needlebench_summary_groups = [
    {'name': 'original_version', 'subsets': _needlebench_4k_origin},
    {'name': 'original_version_zh', 'subsets': _needlebench_4k_origin_zh},
    {'name': 'original_version_en', 'subsets': _needlebench_4k_origin_en},

    {'name': 'multi_needle_en', 'subsets': _needlebench_4k_multi_needle_en},
    {'name': 'multi_needle2_en', 'subsets': _needlebench_4k_2needle_en},
    {'name': 'multi_needle3_en', 'subsets': _needlebench_4k_3needle_en},
    {'name': 'multi_needle4_en', 'subsets': _needlebench_4k_4needle_en},
    {'name': 'multi_needle5_en', 'subsets': _needlebench_4k_5needle_en},

    {'name': 'multi_needle_zh', 'subsets': _needlebench_4k_multi_needle_zh},
    {'name': 'multi_needle2_zh', 'subsets': _needlebench_4k_2needle_zh},
    {'name': 'multi_needle3_zh', 'subsets': _needlebench_4k_3needle_zh},
    {'name': 'multi_needle4_zh', 'subsets': _needlebench_4k_4needle_zh},
    {'name': 'multi_needle5_zh', 'subsets': _needlebench_4k_5needle_zh},

    {'name': 'multi_needle', 'subsets': _needlebench_4k_multi_needle},

    {'name': 'parallel_version', 'subsets': _needlebench_4k_parallel},
    {'name': 'parallel_version_zh', 'subsets': _needlebench_4k_parallel_zh},
    {'name': 'parallel_version_en', 'subsets': _needlebench_4k_parallel_en},


    {'name': 'overall',
     'subsets': [['original_version', 'naive_average'],
                 ['multi_needle', 'naive_average'],
                 ['parallel_version', 'average_score']],
     'weights': {'original_version': 0.4,
                 'multi_needle': 0.3,
                 'parallel_version': 0.3}},
]
needlebench_4k_summarizer = dict(
    type=NeedleBenchSummarizer,
    dataset_abbrs=[
        'overall',
        '--------- NeedleBench-4k Single-Needle ---------',  # category
        'original_version',
        'original_version_zh',
        'original_version_en',
        '--------- NeedleBench-4k Parallel-Needles ---------',  # category
        'parallel_version',
        'parallel_version_zh',
        'parallel_version_en',
        '--------- NeedleBench-4k Multi-Needles ---------',  # category
        'multi_needle',
        'multi_needle_en',
        'multi_needle_zh',
        'multi_needle2_en',
        'multi_needle3_en',
        'multi_needle4_en',
        'multi_needle5_en',
        'multi_needle2_zh',
        'multi_needle3_zh',
        'multi_needle4_zh',
        'multi_needle5_zh',

        # *_needlebench_4k_origin, *_needlebench_4k_multi_needle, *_needlebench_4k_parallel,
    ],
    summary_groups=needlebench_summary_groups,
)

# ----------NeedleBench-8k-summarizer----------

context_lengths_8k = list(range(5000, 9000, 1000))

# Initialize the lists
_needlebench_8k_2needle_en = []
_needlebench_8k_3needle_en = []
_needlebench_8k_4needle_en = []
_needlebench_8k_5needle_en = []
_needlebench_8k_2needle_zh = []
_needlebench_8k_3needle_zh = []
_needlebench_8k_4needle_zh = []
_needlebench_8k_5needle_zh = []
_needlebench_8k_origin_en = []
_needlebench_8k_origin_zh = []

# Fill the lists using nested loops
for original_context_length in context_lengths_8k:
    for depth_percent in depths:
        _needlebench_8k_2needle_en.append(f'Length{original_context_length}Depth{int(depth_percent)}_2needle_en_8k')
        _needlebench_8k_3needle_en.append(f'Length{original_context_length}Depth{int(depth_percent)}_3needle_en_8k')
        _needlebench_8k_4needle_en.append(f'Length{original_context_length}Depth{int(depth_percent)}_4needle_en_8k')
        _needlebench_8k_5needle_en.append(f'Length{original_context_length}Depth{int(depth_percent)}_5needle_en_8k')
        _needlebench_8k_2needle_zh.append(f'Length{original_context_length}Depth{int(depth_percent)}_2needle_zh_8k')
        _needlebench_8k_3needle_zh.append(f'Length{original_context_length}Depth{int(depth_percent)}_3needle_zh_8k')
        _needlebench_8k_4needle_zh.append(f'Length{original_context_length}Depth{int(depth_percent)}_4needle_zh_8k')
        _needlebench_8k_5needle_zh.append(f'Length{original_context_length}Depth{int(depth_percent)}_5needle_zh_8k')

        _needlebench_8k_origin_en.append(f'Length{original_context_length}Depth{int(depth_percent)}_origin_en_8k')
        _needlebench_8k_origin_zh.append(f'Length{original_context_length}Depth{int(depth_percent)}_origin_zh_8k')

# Concatenate the multi-needle and origin lists
_needlebench_8k_multi_needle_en = _needlebench_8k_2needle_en + _needlebench_8k_3needle_en + _needlebench_8k_4needle_en + _needlebench_8k_5needle_en
_needlebench_8k_multi_needle_zh = _needlebench_8k_2needle_zh + _needlebench_8k_3needle_zh + _needlebench_8k_4needle_zh + _needlebench_8k_5needle_zh
_needlebench_8k_origin = _needlebench_8k_origin_en + _needlebench_8k_origin_zh
_needlebench_8k_multi_needle = _needlebench_8k_multi_needle_en + _needlebench_8k_multi_needle_zh

# Repeating the same process for parallel (assuming it's similar to origin_en)
_needlebench_8k_parallel_en = []
_needlebench_8k_parallel_zh = []
for original_context_length in context_lengths_8k:
    _needlebench_8k_parallel_en.append(f'Length{original_context_length}_parallel_en_8k')
for original_context_length in context_lengths_8k:
    _needlebench_8k_parallel_zh.append(f'Length{original_context_length}_parallel_zh_8k')
_needlebench_8k_parallel = _needlebench_8k_parallel_en + _needlebench_8k_parallel_zh

needlebench_summary_groups = [
    {'name': 'original_version', 'subsets': _needlebench_8k_origin},
    {'name': 'original_version_zh', 'subsets': _needlebench_8k_origin_zh},
    {'name': 'original_version_en', 'subsets': _needlebench_8k_origin_en},

    {'name': 'multi_needle_en', 'subsets': _needlebench_8k_multi_needle_en},
    {'name': 'multi_needle2_en', 'subsets': _needlebench_8k_2needle_en},
    {'name': 'multi_needle3_en', 'subsets': _needlebench_8k_3needle_en},
    {'name': 'multi_needle4_en', 'subsets': _needlebench_8k_4needle_en},
    {'name': 'multi_needle5_en', 'subsets': _needlebench_8k_5needle_en},

    {'name': 'multi_needle_zh', 'subsets': _needlebench_8k_multi_needle_zh},
    {'name': 'multi_needle2_zh', 'subsets': _needlebench_8k_2needle_zh},
    {'name': 'multi_needle3_zh', 'subsets': _needlebench_8k_3needle_zh},
    {'name': 'multi_needle4_zh', 'subsets': _needlebench_8k_4needle_zh},
    {'name': 'multi_needle5_zh', 'subsets': _needlebench_8k_5needle_zh},

    {'name': 'multi_needle', 'subsets': _needlebench_8k_multi_needle},

    {'name': 'parallel_version', 'subsets': _needlebench_8k_parallel},
    {'name': 'parallel_version_zh', 'subsets': _needlebench_8k_parallel_zh},
    {'name': 'parallel_version_en', 'subsets': _needlebench_8k_parallel_en},


    {'name': 'overall',
     'subsets': [['original_version', 'naive_average'],
                 ['multi_needle', 'naive_average'],
                 ['parallel_version', 'average_score']],
     'weights': {'original_version': 0.4,
                 'multi_needle': 0.3,
                 'parallel_version': 0.3}},
]
needlebench_8k_summarizer = dict(
    type=NeedleBenchSummarizer,
    dataset_abbrs=[
        'overall',
        '--------- NeedleBench-8k Single-Needle ---------',  # category
        'original_version',
        'original_version_zh',
        'original_version_en',
        '--------- NeedleBench-8k Parallel-Needles ---------',  # category
        'parallel_version',
        'parallel_version_zh',
        'parallel_version_en',
        '--------- NeedleBench-8k Multi-Needles ---------',  # category
        'multi_needle',
        'multi_needle_en',
        'multi_needle_zh',
        'multi_needle2_en',
        'multi_needle3_en',
        'multi_needle4_en',
        'multi_needle5_en',
        'multi_needle2_zh',
        'multi_needle3_zh',
        'multi_needle4_zh',
        'multi_needle5_zh',

        # *_needlebench_8k_origin, *_needlebench_8k_multi_needle, *_needlebench_8k_parallel,
    ],
    summary_groups=needlebench_summary_groups,
)

# ----------NeedleBench-32k-summarizer----------

context_lengths_32k = [9000, 13000, 17000, 21000, 25000, 29000, 31000, 32000]

# Initialize the lists
_needlebench_32k_2needle_en = []
_needlebench_32k_3needle_en = []
_needlebench_32k_4needle_en = []
_needlebench_32k_5needle_en = []
_needlebench_32k_2needle_zh = []
_needlebench_32k_3needle_zh = []
_needlebench_32k_4needle_zh = []
_needlebench_32k_5needle_zh = []
_needlebench_32k_origin_en = []
_needlebench_32k_origin_zh = []

# Fill the lists using nested loops
for original_context_length in context_lengths_32k:
    for depth_percent in depths_list_sparse:
        _needlebench_32k_2needle_en.append(f'Length{original_context_length}Depth{int(depth_percent)}_2needle_en_32k')
        _needlebench_32k_3needle_en.append(f'Length{original_context_length}Depth{int(depth_percent)}_3needle_en_32k')
        _needlebench_32k_4needle_en.append(f'Length{original_context_length}Depth{int(depth_percent)}_4needle_en_32k')
        _needlebench_32k_5needle_en.append(f'Length{original_context_length}Depth{int(depth_percent)}_5needle_en_32k')
        _needlebench_32k_2needle_zh.append(f'Length{original_context_length}Depth{int(depth_percent)}_2needle_zh_32k')
        _needlebench_32k_3needle_zh.append(f'Length{original_context_length}Depth{int(depth_percent)}_3needle_zh_32k')
        _needlebench_32k_4needle_zh.append(f'Length{original_context_length}Depth{int(depth_percent)}_4needle_zh_32k')
        _needlebench_32k_5needle_zh.append(f'Length{original_context_length}Depth{int(depth_percent)}_5needle_zh_32k')

        _needlebench_32k_origin_en.append(f'Length{original_context_length}Depth{int(depth_percent)}_origin_en_32k')
        _needlebench_32k_origin_zh.append(f'Length{original_context_length}Depth{int(depth_percent)}_origin_zh_32k')

# Concatenate the multi-needle and origin lists
_needlebench_32k_multi_needle_en = _needlebench_32k_2needle_en + _needlebench_32k_3needle_en + _needlebench_32k_4needle_en + _needlebench_32k_5needle_en
_needlebench_32k_multi_needle_zh = _needlebench_32k_2needle_zh + _needlebench_32k_3needle_zh + _needlebench_32k_4needle_zh + _needlebench_32k_5needle_zh
_needlebench_32k_origin = _needlebench_32k_origin_en + _needlebench_32k_origin_zh
_needlebench_32k_multi_needle = _needlebench_32k_multi_needle_en + _needlebench_32k_multi_needle_zh

# Repeating the same process for parallel (assuming it's similar to origin_en)
_needlebench_32k_parallel_en = []
_needlebench_32k_parallel_zh = []
for original_context_length in context_lengths_32k:
    _needlebench_32k_parallel_en.append(f'Length{original_context_length}_parallel_en_32k')
for original_context_length in context_lengths_32k:
    _needlebench_32k_parallel_zh.append(f'Length{original_context_length}_parallel_zh_32k')
_needlebench_32k_parallel = _needlebench_32k_parallel_en + _needlebench_32k_parallel_zh

needlebench_summary_groups = [
    {'name': 'original_version', 'subsets': _needlebench_32k_origin},
    {'name': 'original_version_zh', 'subsets': _needlebench_32k_origin_zh},
    {'name': 'original_version_en', 'subsets': _needlebench_32k_origin_en},

    {'name': 'multi_needle_en', 'subsets': _needlebench_32k_multi_needle_en},
    {'name': 'multi_needle2_en', 'subsets': _needlebench_32k_2needle_en},
    {'name': 'multi_needle3_en', 'subsets': _needlebench_32k_3needle_en},
    {'name': 'multi_needle4_en', 'subsets': _needlebench_32k_4needle_en},
    {'name': 'multi_needle5_en', 'subsets': _needlebench_32k_5needle_en},

    {'name': 'multi_needle_zh', 'subsets': _needlebench_32k_multi_needle_zh},
    {'name': 'multi_needle2_zh', 'subsets': _needlebench_32k_2needle_zh},
    {'name': 'multi_needle3_zh', 'subsets': _needlebench_32k_3needle_zh},
    {'name': 'multi_needle4_zh', 'subsets': _needlebench_32k_4needle_zh},
    {'name': 'multi_needle5_zh', 'subsets': _needlebench_32k_5needle_zh},

    {'name': 'multi_needle', 'subsets': _needlebench_32k_multi_needle},

    {'name': 'parallel_version', 'subsets': _needlebench_32k_parallel},
    {'name': 'parallel_version_zh', 'subsets': _needlebench_32k_parallel_zh},
    {'name': 'parallel_version_en', 'subsets': _needlebench_32k_parallel_en},


    {'name': 'overall',
     'subsets': [['original_version', 'naive_average'],
                 ['multi_needle', 'naive_average'],
                 ['parallel_version', 'average_score']],
     'weights': {'original_version': 0.4,
                 'multi_needle': 0.3,
                 'parallel_version': 0.3}},
]
needlebench_32k_summarizer = dict(
    type=NeedleBenchSummarizer,
    dataset_abbrs=[
        'overall',
        '--------- NeedleBench-32k Single-Needle ---------',  # category
        'original_version',
        'original_version_zh',
        'original_version_en',
        '--------- NeedleBench-32k Parallel-Needles ---------',  # category
        'parallel_version',
        'parallel_version_zh',
        'parallel_version_en',
        '--------- NeedleBench-32k Multi-Needles ---------',  # category
        'multi_needle',
        'multi_needle_en',
        'multi_needle_zh',
        'multi_needle2_en',
        'multi_needle3_en',
        'multi_needle4_en',
        'multi_needle5_en',
        'multi_needle2_zh',
        'multi_needle3_zh',
        'multi_needle4_zh',
        'multi_needle5_zh',

        # *_needlebench_32k_origin, *_needlebench_32k_multi_needle, *_needlebench_32k_parallel,
    ],
    summary_groups=needlebench_summary_groups,
)

# ----------NeedleBench-128k-summarizer----------

context_lengths_128k = list([16000, 32000, 48000, 64000, 80000, 96000, 112000, 128000])

# Initialize the lists
_needlebench_128k_2needle_en = []
_needlebench_128k_3needle_en = []
_needlebench_128k_4needle_en = []
_needlebench_128k_5needle_en = []
_needlebench_128k_2needle_zh = []
_needlebench_128k_3needle_zh = []
_needlebench_128k_4needle_zh = []
_needlebench_128k_5needle_zh = []
_needlebench_128k_origin_en = []
_needlebench_128k_origin_zh = []

# Fill the lists using nested loops
for original_context_length in context_lengths_128k:
    for depth_percent in depths_list_sparse:
        _needlebench_128k_2needle_en.append(f'Length{original_context_length}Depth{int(depth_percent)}_2needle_en_128k')
        _needlebench_128k_3needle_en.append(f'Length{original_context_length}Depth{int(depth_percent)}_3needle_en_128k')
        _needlebench_128k_4needle_en.append(f'Length{original_context_length}Depth{int(depth_percent)}_4needle_en_128k')
        _needlebench_128k_5needle_en.append(f'Length{original_context_length}Depth{int(depth_percent)}_5needle_en_128k')
        _needlebench_128k_2needle_zh.append(f'Length{original_context_length}Depth{int(depth_percent)}_2needle_zh_128k')
        _needlebench_128k_3needle_zh.append(f'Length{original_context_length}Depth{int(depth_percent)}_3needle_zh_128k')
        _needlebench_128k_4needle_zh.append(f'Length{original_context_length}Depth{int(depth_percent)}_4needle_zh_128k')
        _needlebench_128k_5needle_zh.append(f'Length{original_context_length}Depth{int(depth_percent)}_5needle_zh_128k')

        _needlebench_128k_origin_en.append(f'Length{original_context_length}Depth{int(depth_percent)}_origin_en_128k')
        _needlebench_128k_origin_zh.append(f'Length{original_context_length}Depth{int(depth_percent)}_origin_zh_128k')

# Concatenate the multi-needle and origin lists
_needlebench_128k_multi_needle_en = _needlebench_128k_2needle_en + _needlebench_128k_3needle_en + _needlebench_128k_4needle_en + _needlebench_128k_5needle_en
_needlebench_128k_multi_needle_zh = _needlebench_128k_2needle_zh + _needlebench_128k_3needle_zh + _needlebench_128k_4needle_zh + _needlebench_128k_5needle_zh
_needlebench_128k_origin = _needlebench_128k_origin_en + _needlebench_128k_origin_zh
_needlebench_128k_multi_needle = _needlebench_128k_multi_needle_en + _needlebench_128k_multi_needle_zh

# Repeating the same process for parallel (assuming it's similar to origin_en)
_needlebench_128k_parallel_en = []
_needlebench_128k_parallel_zh = []
for original_context_length in context_lengths_128k:
    _needlebench_128k_parallel_en.append(f'Length{original_context_length}_parallel_en_128k')
for original_context_length in context_lengths_128k:
    _needlebench_128k_parallel_zh.append(f'Length{original_context_length}_parallel_zh_128k')
_needlebench_128k_parallel = _needlebench_128k_parallel_en + _needlebench_128k_parallel_zh

needlebench_summary_groups = [
    {'name': 'original_version', 'subsets': _needlebench_128k_origin},
    {'name': 'original_version_zh', 'subsets': _needlebench_128k_origin_zh},
    {'name': 'original_version_en', 'subsets': _needlebench_128k_origin_en},

    {'name': 'multi_needle_en', 'subsets': _needlebench_128k_multi_needle_en},
    {'name': 'multi_needle2_en', 'subsets': _needlebench_128k_2needle_en},
    {'name': 'multi_needle3_en', 'subsets': _needlebench_128k_3needle_en},
    {'name': 'multi_needle4_en', 'subsets': _needlebench_128k_4needle_en},
    {'name': 'multi_needle5_en', 'subsets': _needlebench_128k_5needle_en},

    {'name': 'multi_needle_zh', 'subsets': _needlebench_128k_multi_needle_zh},
    {'name': 'multi_needle2_zh', 'subsets': _needlebench_128k_2needle_zh},
    {'name': 'multi_needle3_zh', 'subsets': _needlebench_128k_3needle_zh},
    {'name': 'multi_needle4_zh', 'subsets': _needlebench_128k_4needle_zh},
    {'name': 'multi_needle5_zh', 'subsets': _needlebench_128k_5needle_zh},

    {'name': 'multi_needle', 'subsets': _needlebench_128k_multi_needle},

    {'name': 'parallel_version', 'subsets': _needlebench_128k_parallel},
    {'name': 'parallel_version_zh', 'subsets': _needlebench_128k_parallel_zh},
    {'name': 'parallel_version_en', 'subsets': _needlebench_128k_parallel_en},


    {'name': 'overall',
     'subsets': [['original_version', 'naive_average'],
                 ['multi_needle', 'naive_average'],
                 ['parallel_version', 'average_score']],
     'weights': {'original_version': 0.4,
                 'multi_needle': 0.3,
                 'parallel_version': 0.3}},
]
needlebench_128k_summarizer = dict(
    type=NeedleBenchSummarizer,
    dataset_abbrs=[
        'overall',
        '--------- NeedleBench-128k Single-Needle ---------',  # category
        'original_version',
        'original_version_zh',
        'original_version_en',
        '--------- NeedleBench-128k Parallel-Needles ---------',  # category
        'parallel_version',
        'parallel_version_zh',
        'parallel_version_en',
        '--------- NeedleBench-128k Multi-Needles ---------',  # category
        'multi_needle',
        'multi_needle_en',
        'multi_needle_zh',
        'multi_needle2_en',
        'multi_needle3_en',
        'multi_needle4_en',
        'multi_needle5_en',
        'multi_needle2_zh',
        'multi_needle3_zh',
        'multi_needle4_zh',
        'multi_needle5_zh',

        # *_needlebench_128k_origin, *_needlebench_128k_multi_needle, *_needlebench_128k_parallel,
    ],
    summary_groups=needlebench_summary_groups,
)

# ----------NeedleBench-200k-summarizer----------

context_lengths_200k = list([16000, 48000, 80000, 112000, 128000, 144000, 176000, 200000])
# Initialize the lists
_needlebench_200k_2needle_en = []
_needlebench_200k_3needle_en = []
_needlebench_200k_4needle_en = []
_needlebench_200k_5needle_en = []
_needlebench_200k_2needle_zh = []
_needlebench_200k_3needle_zh = []
_needlebench_200k_4needle_zh = []
_needlebench_200k_5needle_zh = []
_needlebench_200k_origin_en = []
_needlebench_200k_origin_zh = []

# Fill the lists using nested loops
for original_context_length in context_lengths_200k:
    for depth_percent in depths_list_sparse:
        _needlebench_200k_2needle_en.append(f'Length{original_context_length}Depth{int(depth_percent)}_2needle_en_200k')
        _needlebench_200k_3needle_en.append(f'Length{original_context_length}Depth{int(depth_percent)}_3needle_en_200k')
        _needlebench_200k_4needle_en.append(f'Length{original_context_length}Depth{int(depth_percent)}_4needle_en_200k')
        _needlebench_200k_5needle_en.append(f'Length{original_context_length}Depth{int(depth_percent)}_5needle_en_200k')
        _needlebench_200k_2needle_zh.append(f'Length{original_context_length}Depth{int(depth_percent)}_2needle_zh_200k')
        _needlebench_200k_3needle_zh.append(f'Length{original_context_length}Depth{int(depth_percent)}_3needle_zh_200k')
        _needlebench_200k_4needle_zh.append(f'Length{original_context_length}Depth{int(depth_percent)}_4needle_zh_200k')
        _needlebench_200k_5needle_zh.append(f'Length{original_context_length}Depth{int(depth_percent)}_5needle_zh_200k')

        _needlebench_200k_origin_en.append(f'Length{original_context_length}Depth{int(depth_percent)}_origin_en_200k')
        _needlebench_200k_origin_zh.append(f'Length{original_context_length}Depth{int(depth_percent)}_origin_zh_200k')

# Concatenate the multi-needle and origin lists
_needlebench_200k_multi_needle_en = _needlebench_200k_2needle_en + _needlebench_200k_3needle_en + _needlebench_200k_4needle_en + _needlebench_200k_5needle_en
_needlebench_200k_multi_needle_zh = _needlebench_200k_2needle_zh + _needlebench_200k_3needle_zh + _needlebench_200k_4needle_zh + _needlebench_200k_5needle_zh
_needlebench_200k_origin = _needlebench_200k_origin_en + _needlebench_200k_origin_zh
_needlebench_200k_multi_needle = _needlebench_200k_multi_needle_en + _needlebench_200k_multi_needle_zh

# Repeating the same process for parallel (assuming it's similar to origin_en)
_needlebench_200k_parallel_en = []
_needlebench_200k_parallel_zh = []
for original_context_length in context_lengths_200k:
    _needlebench_200k_parallel_en.append(f'Length{original_context_length}_parallel_en_200k')
for original_context_length in context_lengths_200k:
    _needlebench_200k_parallel_zh.append(f'Length{original_context_length}_parallel_zh_200k')
_needlebench_200k_parallel = _needlebench_200k_parallel_en + _needlebench_200k_parallel_zh

needlebench_summary_groups = [
    {'name': 'original_version', 'subsets': _needlebench_200k_origin},
    {'name': 'original_version_zh', 'subsets': _needlebench_200k_origin_zh},
    {'name': 'original_version_en', 'subsets': _needlebench_200k_origin_en},

    {'name': 'multi_needle_en', 'subsets': _needlebench_200k_multi_needle_en},
    {'name': 'multi_needle2_en', 'subsets': _needlebench_200k_2needle_en},
    {'name': 'multi_needle3_en', 'subsets': _needlebench_200k_3needle_en},
    {'name': 'multi_needle4_en', 'subsets': _needlebench_200k_4needle_en},
    {'name': 'multi_needle5_en', 'subsets': _needlebench_200k_5needle_en},

    {'name': 'multi_needle_zh', 'subsets': _needlebench_200k_multi_needle_zh},
    {'name': 'multi_needle2_zh', 'subsets': _needlebench_200k_2needle_zh},
    {'name': 'multi_needle3_zh', 'subsets': _needlebench_200k_3needle_zh},
    {'name': 'multi_needle4_zh', 'subsets': _needlebench_200k_4needle_zh},
    {'name': 'multi_needle5_zh', 'subsets': _needlebench_200k_5needle_zh},

    {'name': 'multi_needle', 'subsets': _needlebench_200k_multi_needle},

    {'name': 'parallel_version', 'subsets': _needlebench_200k_parallel},
    {'name': 'parallel_version_zh', 'subsets': _needlebench_200k_parallel_zh},
    {'name': 'parallel_version_en', 'subsets': _needlebench_200k_parallel_en},

    {'name': 'overall',
     'subsets': [['original_version', 'naive_average'],
                 ['multi_needle', 'naive_average'],
                 ['parallel_version', 'average_score']],
     'weights': {'original_version': 0.4,
                 'multi_needle': 0.3,
                 'parallel_version': 0.3}},
]
needlebench_200k_summarizer = dict(
    type=NeedleBenchSummarizer,
    dataset_abbrs=[
        'overall',
        '--------- NeedleBench-200k Single-Needle ---------',  # category
        'original_version',
        'original_version_zh',
        'original_version_en',
        '--------- NeedleBench-200k Parallel-Needles ---------',  # category
        'parallel_version',
        'parallel_version_zh',
        'parallel_version_en',
        '--------- NeedleBench-200k Multi-Needles ---------',  # category
        'multi_needle',
        'multi_needle_en',
        'multi_needle_zh',
        'multi_needle2_en',
        'multi_needle3_en',
        'multi_needle4_en',
        'multi_needle5_en',
        'multi_needle2_zh',
        'multi_needle3_zh',
        'multi_needle4_zh',
        'multi_needle5_zh',

        # *_needlebench_200k_origin, *_needlebench_200k_multi_needle, *_needlebench_200k_parallel,
    ],
    summary_groups=needlebench_summary_groups,
)
context_lengths_8k = list(range(5000, 9000, 1000))

# Repeating the same process for parallel (assuming it's similar to origin_en)
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
    {'name': 'parallel_version_batch1', 'subsets': [[_dataset, "average_score"] for _dataset in _needlebench_8k_parallel_batch1]},
    {'name': 'parallel_version_zh_batch1', 'subsets': [[_dataset, "average_score"] for _dataset in _needlebench_8k_parallel_zh_batch1]},
    {'name': 'parallel_version_en_batch1', 'subsets': [[_dataset, "average_score"] for _dataset in _needlebench_8k_parallel_en_batch1]},
    {'name': 'parallel_version_batch5', 'subsets': [[_dataset, "average_score"] for _dataset in _needlebench_8k_parallel_batch5]},
    {'name': 'parallel_version_zh_batch5', 'subsets': [[_dataset, "average_score"] for _dataset in _needlebench_8k_parallel_zh_batch5]},
    {'name': 'parallel_version_en_batch5', 'subsets': [[_dataset, "average_score"] for _dataset in _needlebench_8k_parallel_en_batch5]},
    {'name': 'parallel_version_batch10', 'subsets': [[_dataset, "average_score"] for _dataset in _needlebench_8k_parallel_batch10]},
    {'name': 'parallel_version_zh_batch10', 'subsets': [[_dataset, "average_score"] for _dataset in _needlebench_8k_parallel_zh_batch10]},
    {'name': 'parallel_version_en_batch10', 'subsets': [[_dataset, "average_score"] for _dataset in _needlebench_8k_parallel_en_batch10]},
    {'name': 'parallel_version_batch15', 'subsets': [[_dataset, "average_score"] for _dataset in _needlebench_8k_parallel_batch15]},
    {'name': 'parallel_version_zh_batch15', 'subsets': [[_dataset, "average_score"] for _dataset in _needlebench_8k_parallel_zh_batch15]},
    {'name': 'parallel_version_en_batch15', 'subsets': [[_dataset, "average_score"] for _dataset in _needlebench_8k_parallel_en_batch15]},
    {'name': 'parallel_version_batch20', 'subsets': [[_dataset, "average_score"] for _dataset in _needlebench_8k_parallel_batch20]},
    {'name': 'parallel_version_zh_batch20', 'subsets': [[_dataset, "average_score"] for _dataset in _needlebench_8k_parallel_zh_batch20]},
    {'name': 'parallel_version_en_batch20', 'subsets': [[_dataset, "average_score"] for _dataset in _needlebench_8k_parallel_en_batch20]},
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
        # *_needlebench_8k_origin, *_needlebench_8k_multi_needle, *_needlebench_8k_parallel,
    ],
    summary_groups=needlebench_summary_groups,
)

needlebench_summary_groups = [
    {'name': 'parallel_version_batch1', 'subsets': [[_dataset, "Depth0"] for _dataset in _needlebench_8k_parallel_batch1]},
    {'name': 'parallel_version_zh_batch1', 'subsets': [[_dataset, "Depth0"] for _dataset in _needlebench_8k_parallel_zh_batch1]},
    {'name': 'parallel_version_en_batch1', 'subsets': [[_dataset, "Depth0"] for _dataset in _needlebench_8k_parallel_en_batch1]},
    {'name': 'parallel_version_batch5', 'subsets': [[_dataset, "Depth0"] for _dataset in _needlebench_8k_parallel_batch5]},
    {'name': 'parallel_version_zh_batch5', 'subsets': [[_dataset, "Depth0"] for _dataset in _needlebench_8k_parallel_zh_batch5]},
    {'name': 'parallel_version_en_batch5', 'subsets': [[_dataset, "Depth0"] for _dataset in _needlebench_8k_parallel_en_batch5]},
    {'name': 'parallel_version_batch10', 'subsets': [[_dataset, "Depth0"] for _dataset in _needlebench_8k_parallel_batch10]},
    {'name': 'parallel_version_zh_batch10', 'subsets': [[_dataset, "Depth0"] for _dataset in _needlebench_8k_parallel_zh_batch10]},
    {'name': 'parallel_version_en_batch10', 'subsets': [[_dataset, "Depth0"] for _dataset in _needlebench_8k_parallel_en_batch10]},
    {'name': 'parallel_version_batch15', 'subsets': [[_dataset, "Depth0"] for _dataset in _needlebench_8k_parallel_batch15]},
    {'name': 'parallel_version_zh_batch15', 'subsets': [[_dataset, "Depth0"] for _dataset in _needlebench_8k_parallel_zh_batch15]},
    {'name': 'parallel_version_en_batch15', 'subsets': [[_dataset, "Depth0"] for _dataset in _needlebench_8k_parallel_en_batch15]},
    {'name': 'parallel_version_batch20', 'subsets': [[_dataset, "Depth0"] for _dataset in _needlebench_8k_parallel_batch20]},
    {'name': 'parallel_version_zh_batch20', 'subsets': [[_dataset, "Depth0"] for _dataset in _needlebench_8k_parallel_zh_batch20]},
    {'name': 'parallel_version_en_batch20', 'subsets': [[_dataset, "Depth0"] for _dataset in _needlebench_8k_parallel_en_batch20]},
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
        # *_needlebench_8k_origin, *_needlebench_8k_multi_needle, *_needlebench_8k_parallel,
    ],
    summary_groups=needlebench_summary_groups,
)
needlebench_atc_summarizer = dict(
    type=NeedleBenchATCSummarizer,
)
