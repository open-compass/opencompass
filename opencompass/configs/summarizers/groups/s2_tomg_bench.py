subtasks = [
    'MolCustom_AtomNum',
    'MolCustom_BondNum',
    'MolCustom_FunctionalGroup',
    'MolEdit_AddComponent',
    'MolEdit_DelComponent',
    'MolEdit_SubComponent',
    'MolOpt_LogP',
    'MolOpt_MR',
    'MolOpt_QED',
]

s2_tomg_bench_summary_groups = [
    {
        'name': 'S2-TOMG-Bench',
        'subsets': [f'S2-TOMG-Bench-{t}' for t in subtasks],
    },
    {
        'name': 'S2-TOMG-Bench-mini',
        'subsets': [f'S2-TOMG-Bench-{t}-mini' for t in subtasks],
    },
]
