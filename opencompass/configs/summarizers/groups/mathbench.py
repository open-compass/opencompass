from copy import deepcopy

naive_mathbench_summary_groups = [
    {
        'name': 'mathbench-college',
        'subsets': [
            ['mathbench-college-single_choice_cn', 'acc_1'],
            ['mathbench-college-cloze_en', 'accuracy'],
        ]
    },
    {
        'name': 'mathbench-high',
        'subsets': [
            ['mathbench-high-single_choice_cn', 'acc_1'],
            ['mathbench-high-single_choice_en', 'acc_1'],
        ]
    },
    {
        'name': 'mathbench-middle',
        'subsets': [
            ['mathbench-middle-single_choice_cn', 'acc_1'],
        ]
    },
    {
        'name': 'mathbench-primary',
        'subsets': [
            ['mathbench-primary-cloze_cn', 'accuracy'],
        ]
    },
    {
        'name': 'mathbench',
        'subsets': [
            'mathbench-college',
            'mathbench-high',
            'mathbench-middle',
            'mathbench-primary',
        ],
    },
    {
        'name': 'mathbench-college-circular',
        'subsets': [
            ['mathbench-college-single_choice_cn', 'perf_4'],
        ]
    },
    {
        'name': 'mathbench-high-circular',
        'subsets': [
            ['mathbench-high-single_choice_cn', 'perf_4'],
            ['mathbench-high-single_choice_en', 'perf_4'],
        ]
    },
    {
        'name': 'mathbench-middle-circular',
        'subsets': [
            ['mathbench-middle-single_choice_cn', 'perf_4'],
        ]
    },
    {
        'name': 'mathbench-circular',
        'subsets': [
            'mathbench-college-circular',
            'mathbench-high-circular',
            'mathbench-middle-circular',
        ],
    },
    {
        'name': 'mathbench-circular-and-cloze',
        'subsets': [
            'mathbench-college-circular',
            'mathbench-high-circular',
            'mathbench-middle-circular',
            'mathbench-college-cloze_en',
            'mathbench-primary-cloze_cn',
        ],
    }
]

agent_mathbench_summary_groups = []
for item in naive_mathbench_summary_groups:
    item = deepcopy(item)
    item['name'] = item['name'] + '-agent'
    if isinstance(item['subsets'][0], str):
        item['subsets'] = [i + '-agent' for i in item['subsets']]
    else:
        item['subsets'] = [[i[0] + '-agent', i[1]] for i in item['subsets']]
    agent_mathbench_summary_groups.append(item)

mathbench_summary_groups = naive_mathbench_summary_groups + agent_mathbench_summary_groups
