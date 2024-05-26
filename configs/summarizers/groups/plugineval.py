from copy import deepcopy

_base_summary_groups = [
    {
        'name': 'plugin_eval-instruct_v1',
        'metric': 'format_metric',
        'subsets': [
            ['plugin_eval-instruct_v1', 'string_format_metric'],
            ['plugin_eval-instruct_v1', 'json_format_metric'],
        ]
    },
    {
        'name': 'plugin_eval-instruct_v1',
        'metric': 'args_em_metric',
        'subsets': [
            ['plugin_eval-instruct_v1', 'string_args_em_metric'],
            ['plugin_eval-instruct_v1', 'json_args_em_metric'],
        ]
    },
    {
        'name': 'plugin_eval-instruct_v1',
        'metric': 'string_metric',
        'subsets': [
            ['plugin_eval-instruct_v1', 'string_format_metric'],
            ['plugin_eval-instruct_v1', 'string_args_em_metric'],
        ]
    },
    {
        'name': 'plugin_eval-instruct_v1',
        'metric': 'json_metric',
        'subsets': [
            ['plugin_eval-instruct_v1', 'json_format_metric'],
            ['plugin_eval-instruct_v1', 'json_args_em_metric'],
        ]
    },
    {
        'name': 'copy_plugin_eval-review_str_v1',
        'subsets': [
            ['plugin_eval-review_str_v1', 'review_quality'],
        ],
    },
    {
        'name': 'plugin_eval_one_review',
        'subsets': [
            ['plugin_eval-instruct_v1', 'format_metric'],
            ['plugin_eval-instruct_v1', 'args_em_metric'],
            ['plugin_eval-plan_str_v1', 'f1_score'],
            ['plugin_eval-plan_json_v1', 'f1_score'],
            ['plugin_eval-reason_str_v1', 'thought'],
            ['plugin_eval-reason_retrieve_understand_json_v1', 'thought'],
            ['plugin_eval-retrieve_str_v1', 'name'],
            ['plugin_eval-reason_retrieve_understand_json_v1', 'name'],
            ['plugin_eval-understand_str_v1', 'args'],
            ['plugin_eval-reason_retrieve_understand_json_v1', 'args'],
            ['plugin_eval-review_str_v1', 'review_quality'],
        ]
    },
    {
        'name': 'plugin_eval',
        'subsets': [
            ['plugin_eval-instruct_v1', 'format_metric'],
            ['plugin_eval-instruct_v1', 'args_em_metric'],
            ['plugin_eval-plan_str_v1', 'f1_score'],
            ['plugin_eval-plan_json_v1', 'f1_score'],
            ['plugin_eval-reason_str_v1', 'thought'],
            ['plugin_eval-reason_retrieve_understand_json_v1', 'thought'],
            ['plugin_eval-retrieve_str_v1', 'name'],
            ['plugin_eval-reason_retrieve_understand_json_v1', 'name'],
            ['plugin_eval-understand_str_v1', 'args'],
            ['plugin_eval-reason_retrieve_understand_json_v1', 'args'],
            ['plugin_eval-review_str_v1', 'review_quality'],
        ]
    },

    # special treatment for first 10% data points
    {
        'name': 'plugin_eval-p10-instruct_v1',
        'metric': 'format_metric',
        'subsets': [
            ['plugin_eval-p10-instruct_v1', 'string_format_metric'],
            ['plugin_eval-p10-instruct_v1', 'json_format_metric'],
        ]
    },
    {
        'name': 'plugin_eval-p10-instruct_v1',
        'metric': 'args_em_metric',
        'subsets': [
            ['plugin_eval-p10-instruct_v1', 'string_args_em_metric'],
            ['plugin_eval-p10-instruct_v1', 'json_args_em_metric'],
        ]
    },
    {
        'name': 'plugin_eval-p10',
        'subsets': [
            ['plugin_eval-p10-instruct_v1', 'format_metric'],
            ['plugin_eval-p10-instruct_v1', 'args_em_metric'],
            ['plugin_eval-p10-plan_str_v1', 'f1_score'],
            ['plugin_eval-p10-plan_json_v1', 'f1_score'],
            ['plugin_eval-p10-reason_str_v2', 'thought'],
            ['plugin_eval-p10-reason_retrieve_understand_json_v2', 'thought'],
            ['plugin_eval-p10-retrieve_str_v2', 'name'],
            ['plugin_eval-p10-reason_retrieve_understand_json_v2', 'name'],
            ['plugin_eval-p10-understand_str_v2', 'args'],
            ['plugin_eval-p10-reason_retrieve_understand_json_v2', 'args'],
            ['plugin_eval-p10-review_str_v6', 'review_quality'],
        ]
    },
]

plugineval_summary_groups = []

# base
for group in _base_summary_groups:
    group = deepcopy(group)
    plugineval_summary_groups.append(group)

# base _zh
for group in _base_summary_groups:
    group = deepcopy(group)
    group['name'] = group['name'] + '_zh'
    group['subsets'] = [[subset[0] + '_zh', subset[1]] for subset in group['subsets']]
    plugineval_summary_groups.append(group)

# base -p10-
for group in _base_summary_groups:
    group = deepcopy(group)
    group['name'] = group['name'].replace('plugin_eval', 'plugin_eval-p10')
    group['subsets'] = [[subset[0].replace('plugin_eval', 'plugin_eval-p10'), subset[1]] for subset in group['subsets']]
    plugineval_summary_groups.append(group)

# base -p10- _zh
for group in _base_summary_groups:
    group = deepcopy(group)
    group['name'] = group['name'].replace('plugin_eval', 'plugin_eval-p10') + '_zh'
    group['subsets'] = [[subset[0].replace('plugin_eval', 'plugin_eval-p10') + '_zh', subset[1]] for subset in group['subsets']]
    plugineval_summary_groups.append(group)

# base -mus-p10-
for group in _base_summary_groups:
    group = deepcopy(group)
    group['name'] = group['name'].replace('plugin_eval', 'plugin_eval-mus-p10')
    group['subsets'] = [[subset[0].replace('plugin_eval', 'plugin_eval-mus-p10'), subset[1]] for subset in group['subsets']]
    plugineval_summary_groups.append(group)

# base -mus-p10- _zh
for group in _base_summary_groups:
    group = deepcopy(group)
    group['name'] = group['name'].replace('plugin_eval', 'plugin_eval-mus-p10') + '_zh'
    group['subsets'] = [[subset[0].replace('plugin_eval', 'plugin_eval-mus-p10') + '_zh', subset[1]] for subset in group['subsets']]
    plugineval_summary_groups.append(group)
