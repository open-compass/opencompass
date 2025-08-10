mathbench_v1_summary_groups = [
    {'name': 'mathbench-college_application', 'subsets': ['mathbench-college-single_choice_cn', 'mathbench-college-single_choice_en']},
    {'name': 'mathbench-high_application', 'subsets': ['mathbench-high-single_choice_cn', 'mathbench-high-single_choice_en']},
    {'name': 'mathbench-middle_application', 'subsets': ['mathbench-middle-single_choice_cn', 'mathbench-middle-single_choice_en']},
    {'name': 'mathbench-primary_application', 'subsets': ['mathbench-primary-cloze_cn', 'mathbench-primary-cloze_en', 'mathbench-calculate-cloze_en'], 'weights': {'mathbench-primary-cloze_cn': 1, 'mathbench-primary-cloze_en': 1, 'mathbench-calculate-cloze_en': 2}},
    {'name': 'mathbench-college_knowledge', 'subsets': ['mathbench-college_knowledge-single_choice_cn', 'mathbench-college_knowledge-single_choice_en']},
    {'name': 'mathbench-high_knowledge', 'subsets': ['mathbench-high_knowledge-single_choice_cn', 'mathbench-high_knowledge-single_choice_en']},
    {'name': 'mathbench-middle_knowledge', 'subsets': ['mathbench-middle_knowledge-single_choice_cn', 'mathbench-middle_knowledge-single_choice_en']},
    {'name': 'mathbench-primary_knowledge', 'subsets': ['mathbench-primary_knowledge-single_choice_cn', 'mathbench-primary_knowledge-single_choice_en']},
    {'name': 'mathbench_application', 'subsets': ['mathbench-college_application', 'mathbench-high_application', 'mathbench-middle_application', 'mathbench-primary_application']},
    {'name': 'mathbench_knowledge', 'subsets': ['mathbench-college_knowledge', 'mathbench-high_knowledge', 'mathbench-middle_knowledge', 'mathbench-primary_knowledge']},
    {'name': 'mathbench', 'subsets': ['mathbench_application', 'mathbench_knowledge']},
]
