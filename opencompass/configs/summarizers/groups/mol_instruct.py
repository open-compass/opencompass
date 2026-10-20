categories = [
    ['FS-selfies', 'score'],
    ['MC-selfies', 'score'],
    ['MG-selfies', 'score'],
    ['PP-selfies', 'score'],
    ['RP-selfies', 'score'],
    ['RS-selfies', 'score'],
]

mol_instruct_summary_groups = [{
    'name': 'mol_instruct',
    'subsets': categories,
    'transforms': {
        'PP-selfies': 'max((3 - x) / 3, 0) * 100',
    },
    },
]