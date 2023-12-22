plugineval_summary_groups = [
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
        'name': 'plugin_eval',
        'subsets': [
            ['plugin_eval-instruct_v1', 'format_metric'],
            ['plugin_eval-instruct_v1', 'args_em_metric'],
            ['plugin_eval-plan_str_v1', 'f1_score'],
            ['plugin_eval-plan_json_v1', 'f1_score'],
            ['plugin_eval-reason_str_v2', 'thought'],
            ['plugin_eval-reason_retrieve_understand_json_v2', 'thought'],
            ['plugin_eval-retrieve_str_v2', 'name'],
            ['plugin_eval-reason_retrieve_understand_json_v2', 'name'],
            ['plugin_eval-understand_str_v2', 'args'],
            ['plugin_eval-reason_retrieve_understand_json_v2', 'args'],
            ['plugin_eval-review_str_v6', 'review_quality'],
        ]
    },
]
