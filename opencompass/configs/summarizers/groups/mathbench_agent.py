
mathbench_agent_summary_groups = [
    {
        'name': 'mathbench-college-agent',
        'subsets': [
            ['mathbench-college-single_choice_cn-agent', 'acc_1'],
            ['mathbench-college-cloze_en-agent', 'accuracy'],
        ]
    },
    {
        'name': 'mathbench-high-agent',
        'subsets': [
            ['mathbench-high-single_choice_cn-agent', 'acc_1'],
            ['mathbench-high-single_choice_en-agent', 'acc_1'],
        ]
    },
    {
        'name': 'mathbench-middle-agent',
        'subsets': [
            ['mathbench-middle-single_choice_cn-agent', 'acc_1'],
        ]
    },
    {
        'name': 'mathbench-primary-agent',
        'subsets': [
            ['mathbench-primary-cloze_cn-agent', 'accuracy'],
        ]
    },
    {
        'name': 'mathbench-agent',
        'subsets': [
            'mathbench-college-agent',
            'mathbench-high-agent',
            'mathbench-middle-agent',
            'mathbench-primary-agent',
        ],
    },
    {
        'name': 'mathbench-college-circular-agent',
        'subsets': [
            ['mathbench-college-single_choice_cn-agent', 'perf_4'],
        ]
    },
    {
        'name': 'mathbench-high-circular-agent',
        'subsets': [
            ['mathbench-high-single_choice_cn-agent', 'perf_4'],
            ['mathbench-high-single_choice_en-agent', 'perf_4'],
        ]
    },
    {
        'name': 'mathbench-middle-circular-agent',
        'subsets': [
            ['mathbench-middle-single_choice_cn-agent', 'perf_4'],
        ]
    },
    {
        'name': 'mathbench-circular-agent',
        'subsets': [
            'mathbench-college-circular-agent',
            'mathbench-high-circular-agent',
            'mathbench-middle-circular-agent',
        ],
    },
    {
        'name': 'mathbench-circular-and-cloze-agent',
        'subsets': [
            'mathbench-college-circular-agent',
            'mathbench-high-circular-agent',
            'mathbench-middle-circular-agent',
            'mathbench-college-cloze_en-agent',
            'mathbench-primary-cloze_cn-agent',
        ],
    }
]
