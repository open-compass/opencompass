"""RULER summary groups."""

default_ruler_tasks = [
    'ruler_niah_single_1',
    'ruler_niah_single_2',
    'ruler_niah_single_3',
    'ruler_niah_multikey_1',
    'ruler_niah_multikey_2',
    'ruler_niah_multikey_3',
    'ruler_niah_multivalue',
    'ruler_niah_multiquery',
    'ruler_vt',
    'ruler_fwe',
    'ruler_cwe',
    'ruler_qa_squad',
    'ruler_qa_hotpotqa',
]
context_window_sizes = ['4k', '8k', '16k', '32k', '64k', '128k', '256k', '512k', '1m']

ruler_summary_groups = []
for context_window_size in context_window_sizes:
    ruler_summary_groups.append(
        {
            'name': f'ruler_{context_window_size}',
            'subsets': [
                f'{task}_{context_window_size}' for task in default_ruler_tasks
            ],
        }
    )
