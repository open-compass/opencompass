
mathbench_2024_summary_groups = [
    # {'name': 'college', 'subsets': ['college-single_choice_cn', 'college-single_choice_en']},
    # {'name': 'high', 'subsets': ['high-single_choice_cn', 'high-single_choice_en']},
    # {'name': 'middle', 'subsets': ['middle-single_choice_cn', 'middle-single_choice_en']},
    # {'name': 'primary', 'subsets': ['primary-cloze_cn', 'primary-cloze_en']},
    {'name': 'college', 'subsets': [['mathbench-college-single_choice_cn', 'perf_4'], ['mathbench-college-single_choice_en', 'perf_4']]},
    {'name': 'high', 'subsets': [['mathbench-high-single_choice_cn', 'perf_4'], ['mathbench-high-single_choice_en', 'perf_4']]},
    {'name': 'middle', 'subsets': [['mathbench-middle-single_choice_cn', 'perf_4'], ['mathbench-middle-single_choice_en', 'perf_4']]},
    {'name': 'primary', 'subsets': [['mathbench-primary-cloze_cn', 'accuracy'], ['mathbench-primary-cloze_en', 'accuracy']]},
    {'name': 'arithmetic', 'subsets': [['mathbench-arithmetic-cloze_en', 'accuracy']]},
    # {'name': 'cn', 'subsets': ['college-single_choice_cn', 'high-single_choice_cn', 'middle-single_choice_cn', 'primary-cloze_cn']},
    # {'name': 'en', 'subsets': ['college-single_choice_en', 'high-single_choice_en', 'middle-single_choice_en', 'primary-cloze_en']},
    {'name': 'mathbench-a-cn', 'subsets': ['mathbench-college-single_choice_cn', 'mathbench-high-single_choice_cn', 'mathbench-middle-single_choice_cn', 'mathbench-primary-cloze_cn']},
    {'name': 'mathbench-a-en', 'subsets': ['mathbench-college-single_choice_en', 'mathbench-high-single_choice_en', 'mathbench-middle-single_choice_en', 'mathbench-primary-cloze_en']},
    {'name': 'mathbench-a (average)', 'subsets': ['college', 'high', 'middle', 'primary', 'arithmetic']},

    # {'name': 'college_knowledge', 'subsets': ['mathbench-college_knowledge-single_choice_cn', 'mathbench-college_knowledge-single_choice_en']},
    # {'name': 'high_knowledge', 'subsets': ['mathbench-high_knowledge-single_choice_cn', 'mathbench-high_knowledge-single_choice_en']},
    # {'name': 'middle_knowledge', 'subsets': ['mathbench-middle_knowledge-single_choice_cn', 'mathbench-middle_knowledge-single_choice_en']},
    # {'name': 'primary_knowledge', 'subsets': ['mathbench-primary_knowledge-single_choice_cn', 'mathbench-primary_knowledge-single_choice_en']},
    # {'name': 'knowledge-cn', 'subsets': ['college_knowledge-single_choice_cn', 'high_knowledge-single_choice_cn', 'middle_knowledge-single_choice_cn', 'primary_knowledge-single_choice_cn']},
    # {'name': 'knowledge-en', 'subsets': ['college_knowledge-single_choice_en', 'high_knowledge-single_choice_en', 'middle_knowledge-single_choice_en', 'primary_knowledge-single_choice_en']},
    {'name': 'college_knowledge', 'subsets': [['mathbench-college_knowledge-single_choice_cn', 'perf_4'], ['mathbench-college_knowledge-single_choice_en', 'perf_4']]},
    {'name': 'high_knowledge', 'subsets': [['mathbench-high_knowledge-single_choice_cn', 'perf_4'], ['mathbench-high_knowledge-single_choice_en', 'perf_4']]},
    {'name': 'middle_knowledge', 'subsets': [['mathbench-middle_knowledge-single_choice_cn', 'perf_4'], ['mathbench-middle_knowledge-single_choice_en', 'perf_4']]},
    {'name': 'primary_knowledge', 'subsets': [['mathbench-primary_knowledge-single_choice_cn', 'perf_4'], ['mathbench-primary_knowledge-single_choice_en', 'perf_4']]},
    {'name': 'mathbench-t-cn', 'subsets': ['mathbench-college_knowledge-single_choice_cn', 'mathbench-high_knowledge-single_choice_cn', 'mathbench-middle_knowledge-single_choice_cn', 'mathbench-primary_knowledge-single_choice_cn']},
    {'name': 'mathbench-t-en', 'subsets': ['mathbench-college_knowledge-single_choice_en', 'mathbench-high_knowledge-single_choice_en', 'mathbench-middle_knowledge-single_choice_en', 'mathbench-primary_knowledge-single_choice_en']},
    {'name': 'mathbench-t (average)', 'subsets': ['college_knowledge', 'high_knowledge', 'middle_knowledge', 'primary_knowledge']},

    {'name': 'Overall', 'subsets': ['mathbench-a (average)', 'mathbench-t (average)']},
]

# for g in mathbench_2024_wocircular_summary_groups:
#     g['name'] = 'mathbench-wocircular-' + g['name']
#     g['subsets'] = ['mathbench-wocircular-' + s for s in g['subsets']]

# mathbench_2024_summary_groups = mathbench_2024_wocircular_summary_groups

summarizer = dict(
    dataset_abbrs = [
        '###### MathBench-A: Application Part ######',
        'college',
        'high',
        'middle',
        'primary',
        'arithmetic',
        'mathbench-a (average)',

        '###### MathBench-T: Theory Part ######',
        'college_knowledge',
        'high_knowledge',
        'middle_knowledge',
        'primary_knowledge',
        'mathbench-t (average)',

        '###### Overall: Average between MathBench-A and MathBench-T ######',
        'Overall',
    ],
    summary_groups=mathbench_2024_summary_groups,
)
