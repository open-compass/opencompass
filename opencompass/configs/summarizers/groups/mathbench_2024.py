
mathbench_2024_wocircular_summary_groups = [
    {'name': 'college', 'subsets': ['college-single_choice_cn', 'college-single_choice_en']},
    {'name': 'high', 'subsets': ['high-single_choice_cn', 'high-single_choice_en']},
    {'name': 'middle', 'subsets': ['middle-single_choice_cn', 'middle-single_choice_en']},
    {'name': 'primary', 'subsets': ['primary-cloze_cn', 'primary-cloze_en']},
    {'name': 'cn', 'subsets': ['college-single_choice_cn', 'high-single_choice_cn', 'middle-single_choice_cn', 'primary-cloze_cn']},
    {'name': 'en', 'subsets': ['college-single_choice_en', 'high-single_choice_en', 'middle-single_choice_en', 'primary-cloze_en']},
    {'name': 'a', 'subsets': ['college', 'high', 'middle', 'primary', 'arithmetic-cloze_en']},

    {'name': 'college_knowledge', 'subsets': ['college_knowledge-single_choice_cn', 'college_knowledge-single_choice_en']},
    {'name': 'high_knowledge', 'subsets': ['high_knowledge-single_choice_cn', 'high_knowledge-single_choice_en']},
    {'name': 'middle_knowledge', 'subsets': ['middle_knowledge-single_choice_cn', 'middle_knowledge-single_choice_en']},
    {'name': 'primary_knowledge', 'subsets': ['primary_knowledge-single_choice_cn', 'primary_knowledge-single_choice_en']},
    {'name': 'knowledge-cn', 'subsets': ['college_knowledge-single_choice_cn', 'high_knowledge-single_choice_cn', 'middle_knowledge-single_choice_cn', 'primary_knowledge-single_choice_cn']},
    {'name': 'knowledge-en', 'subsets': ['college_knowledge-single_choice_en', 'high_knowledge-single_choice_en', 'middle_knowledge-single_choice_en', 'primary_knowledge-single_choice_en']},
    {'name': 't', 'subsets': ['college_knowledge', 'high_knowledge', 'middle_knowledge', 'primary_knowledge']},

    {'name': 'overall', 'subsets': ['a', 't']},
]

for g in mathbench_2024_wocircular_summary_groups:
    g['name'] = 'mathbench-wocircular-' + g['name']
    g['subsets'] = ['mathbench-wocircular-' + s for s in g['subsets']]

mathbench_2024_summary_groups = mathbench_2024_wocircular_summary_groups
