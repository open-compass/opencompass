from copy import deepcopy

_base_summary_groups = [
    {
        'name': 'teval-instruct_v2',
        'metric': 'format_metric',
        'subsets': [
            ['teval-instruct_v2', 'string_format_metric'],
            ['teval-instruct_v2', 'json_format_metric'],
        ]
    },
    {
        'name': 'teval-instruct_v2',
        'metric': 'args_em_metric',
        'subsets': [
            ['teval-instruct_v2', 'string_args_em_metric'],
            ['teval-instruct_v2', 'json_args_em_metric'],
        ]
    },
    {
        'name': 'teval-instruct_v2',
        'metric': 'string_metric',
        'subsets': [
            ['teval-instruct_v2', 'string_format_metric'],
            ['teval-instruct_v2', 'string_args_em_metric'],
        ]
    },
    {
        'name': 'teval-instruct_v2',
        'metric': 'json_metric',
        'subsets': [
            ['teval-instruct_v2', 'json_format_metric'],
            ['teval-instruct_v2', 'json_args_em_metric'],
        ]
    },
    {
        'name': 'copy_teval-review_str_v2',
        'subsets': [
            ['teval-review_str_v2', 'review_quality'],
        ],
    },
    {
        'name': 'teval',
        'subsets': [
            ['teval-instruct_v2', 'format_metric'],
            ['teval-instruct_v2', 'args_em_metric'],
            ['teval-plan_str_v2', 'f1_score'],
            ['teval-plan_json_v2', 'f1_score'],
            ['teval-reason_str_v2', 'thought'],
            ['teval-reason_retrieve_understand_json_v2', 'thought'],
            ['teval-retrieve_str_v2', 'name'],
            ['teval-reason_retrieve_understand_json_v2', 'name'],
            ['teval-understand_str_v2', 'args'],
            ['teval-reason_retrieve_understand_json_v2', 'args'],
            ['teval-review_str_v2', 'review_quality'],
            ['copy_teval-review_str_v2', 'naive_average'],  # a hack for review * 2
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
    group['name'] = group['name'] + '_zh'
    group['subsets'] = [[subset[0] + '_zh', subset[1]] for subset in group['subsets']]
    teval_summary_groups.append(group)
