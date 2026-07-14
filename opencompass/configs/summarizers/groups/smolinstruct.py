categories = [
    ['NC-I2F-0shot-instruct', 'score'],
    ['NC-I2S-0shot-instruct', 'score'],
    ['NC-S2F-0shot-instruct', 'score'],
    ['NC-S2I-0shot-instruct', 'score'],
    ['PP-ESOL-0shot-instruct', 'score'],
    ['PP-Lipo-0shot-instruct', 'score'],
    ['PP-BBBP-0shot-instruct', 'accuracy'],
    ['PP-ClinTox-0shot-instruct', 'accuracy'],
    ['PP-HIV-0shot-instruct', 'accuracy'],
    ['PP-SIDER-0shot-instruct', 'accuracy'],
    ['MC-0shot-instruct', 'score'],
    ['MG-0shot-instruct', 'score'],
    ['FS-0shot-instruct', 'score'],
    ['RS-0shot-instruct', 'score'],
]

mini_categories = [[s[0] + '-mini', s[1]] for s in categories]

smolinstruct_summary_groups = [{
    'name': 'smolinstruct',
    'subsets': categories,
    'transforms': {
        'MC-0shot-instruct': 'x * 100',
        'PP-ESOL-0shot-instruct': 'max((2 - x) / 2, 0) * 100',
        'PP-Lipo-0shot-instruct': 'max((1.2 - x) / 1.2, 0) * 100',
    },
}]

smolinstruct_mini_summary_groups = [{
    'name': 'smolinstruct_mini',
    'subsets': mini_categories,
    'transforms': {
        'MC-0shot-instruct-mini': 'x * 100',
        'PP-ESOL-0shot-instruct-mini': 'max((2 - x) / 2, 0) * 100',
        'PP-Lipo-0shot-instruct-mini': 'max((1.2 - x) / 1.2, 0) * 100',
    },
}]
