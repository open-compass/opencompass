categories = [
    'OE_TO_maths_en_COMP', # OpenEnded - TextOnly - maths - COMP
    'OE_TO_maths_zh_COMP', # OpenEnded - TextOnly - maths - COMP
    'OE_TO_maths_zh_CEE', # OpenEnded - TextOnly - maths - CEE
    'OE_TO_physics_en_COMP', # OpenEnded - TextOnly - physics - COMP
    'OE_TO_physics_zh_CEE' # OpenEnded - TextOnly - physics - CEE
]

OlympiadBench_summary_groups = [
    {'name': 'OlympiadBench', 'subsets': ['OlympiadBench_' + c.replace(' ', '_') for c in categories]},
]
