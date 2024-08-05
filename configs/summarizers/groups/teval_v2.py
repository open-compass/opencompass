from copy import deepcopy

_base_summary_groups = [
    {
        'name': 'teval-instruct_v2_subset',
        'metric': 'format_metric',
        'subsets': [
            ['teval-instruct_v2_subset', 'string_format_metric'],
            ['teval-instruct_v2_subset', 'json_format_metric'],
        ]
    },
    {
        'name': 'teval-instruct_v2_subset',
        'metric': 'args_em_metric',
        'subsets': [
            ['teval-instruct_v2_subset', 'string_args_em_metric'],
            ['teval-instruct_v2_subset', 'json_args_em_metric'],
        ]
    },
    {
        'name': 'teval-instruct_v2_subset',
        'metric': 'string_metric',
        'subsets': [
            ['teval-instruct_v2_subset', 'string_format_metric'],
            ['teval-instruct_v2_subset', 'string_args_em_metric'],
        ]
    },
    {
        'name': 'teval-instruct_v2_subset',
        'metric': 'json_metric',
        'subsets': [
            ['teval-instruct_v2_subset', 'json_format_metric'],
            ['teval-instruct_v2_subset', 'json_args_em_metric'],
        ]
    },
    {
        'name': 'copy_teval-review_str_v2_subset',
        'subsets': [
            ['teval-review_str_v2_subset', 'review_quality'],
        ],
    },
    {
        'name': 'teval',
        'subsets': [
            ['teval-instruct_v2_subset', 'format_metric'],
            ['teval-instruct_v2_subset', 'args_em_metric'],
            ['teval-plan_str_v2_subset', 'f1_score'],
            ['teval-plan_json_v2_subset', 'f1_score'],
            ['teval-reason_str_v2_subset', 'thought'],
            ['teval-reason_retrieve_understand_json_v2_subset', 'thought'],
            ['teval-retrieve_str_v2_subset', 'name'],
            ['teval-reason_retrieve_understand_json_v2_subset', 'name'],
            ['teval-understand_str_v2_subset', 'args'],
            ['teval-reason_retrieve_understand_json_v2_subset', 'args'],
            ['teval-review_str_v2_subset', 'review_quality'],
            ['copy_teval-review_str_v2_subset', 'naive_average'],  # a hack for review * 2
        ]
    },
]

teval_summary_groups = []

# base
for group in _base_summary_groups:
    group = deepcopy(group)
    teval_summary_groups.append(group)

# base _zh
for group in _base_summary_groups:
    group = deepcopy(group)
    group['name'] = group['name'] + '_subset_zh'
    group['subsets'] = [[subset[0] + '_subset_zh', subset[1]] for subset in group['subsets']]
    teval_summary_groups.append(group)
