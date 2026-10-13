inverse_ifeval_instruction_type_abbrs = [
    'QC',
    'ITF',
    'CC',
    'CCF',
    'DIA',
    'II',
    'MIM',
    'CA',
]

inverse_ifeval_language_abbrs = ['zh', 'en']

inverse_ifeval_subsets = [
    f'InverseIFEval_{language}_{instruction_type}'
    for language in inverse_ifeval_language_abbrs
    for instruction_type in inverse_ifeval_instruction_type_abbrs
]

inverse_ifeval_type_counts = {
    'QC': 90,
    'ITF': 86,
    'CC': 198,
    'CCF': 82,
    'DIA': 186,
    'II': 154,
    'MIM': 108,
    'CA': 108,
}

inverse_ifeval_weights = {
    f'InverseIFEval_{language}_{instruction_type}':
    inverse_ifeval_type_counts[instruction_type] // 2
    for language in inverse_ifeval_language_abbrs
    for instruction_type in inverse_ifeval_instruction_type_abbrs
}

inverse_ifeval_summary_groups = [
    dict(
        name='InverseIFEval',
        subsets=[[subset, 'accuracy'] for subset in inverse_ifeval_subsets],
        weights=inverse_ifeval_weights,
    ),
    dict(
        name='InverseIFEval_zh',
        subsets=[[
            f'InverseIFEval_zh_{instruction_type}', 'accuracy'
        ] for instruction_type in inverse_ifeval_instruction_type_abbrs],
        weights={
            f'InverseIFEval_zh_{instruction_type}':
            inverse_ifeval_type_counts[instruction_type] // 2
            for instruction_type in inverse_ifeval_instruction_type_abbrs
        },
    ),
    dict(
        name='InverseIFEval_en',
        subsets=[[
            f'InverseIFEval_en_{instruction_type}', 'accuracy'
        ] for instruction_type in inverse_ifeval_instruction_type_abbrs],
        weights={
            f'InverseIFEval_en_{instruction_type}':
            inverse_ifeval_type_counts[instruction_type] // 2
            for instruction_type in inverse_ifeval_instruction_type_abbrs
        },
    ),
    dict(
        name='InverseIFEval_macro',
        subsets=[[subset, 'accuracy'] for subset in inverse_ifeval_subsets],
    ),
]

inverse_ifeval_summary_groups += [
    dict(
        name=f'InverseIFEval_{instruction_type}',
        subsets=[[
            f'InverseIFEval_{language}_{instruction_type}', 'accuracy'
        ] for language in inverse_ifeval_language_abbrs],
    ) for instruction_type in inverse_ifeval_instruction_type_abbrs
]
