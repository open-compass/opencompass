default_babilong_tasks = [
    'qa1',
    'qa2',
    'qa3',
    'qa4',
    'qa5',
    'qa6',
    'qa7',
    'qa8',
    'qa9',
    'qa10',
]
context_window_sizes = [
    '0k',
    '1k',
    '2k',
    '4k',
    '8k',
    '16k',
    '32k',
    '64k',
    '128k',
    '256k',
    '512k',
    '1m',
]
babilong_summary_groups = []
for context_window_size in context_window_sizes:
    babilong_summary_groups.append(
        {
            'name': f'babilong_{context_window_size}',
            'subsets': [
                f'babilong_{task}_{context_window_size}'
                for task in default_babilong_tasks
            ],
        }
    )
