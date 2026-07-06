categories = [
    ['matbench_expt_gap', 'mae'],
    ['matbench_steels', 'mae'],
    ['matbench_expt_is_metal', 'accuracy'],
    ['matbench_glass', 'accuracy'],
]

matbench_summary_groups = [{
    'name': 'matbench',
    'subsets': categories,
    'transforms': {
        'matbench_expt_gap': '((2 - x) / 2) * 100',
        'matbench_steels': '((2000 - x) / 2000) * 100',
        'matbench_expt_is_metal': 'x',
        'matbench_glass': 'x',
    },
    },
]