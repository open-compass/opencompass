
mathbench_2024_summary_groups = [
    {'name': 'college', 'subsets': [['mathbench-college-single_choice_cn', 'perf_4'], ['mathbench-college-single_choice_en', 'perf_4']]},
    {'name': 'high', 'subsets': [['mathbench-high-single_choice_cn', 'perf_4'], ['mathbench-high-single_choice_en', 'perf_4']]},
    {'name': 'middle', 'subsets': [['mathbench-middle-single_choice_cn', 'perf_4'], ['mathbench-middle-single_choice_en', 'perf_4']]},
    {'name': 'primary', 'subsets': [['mathbench-primary-cloze_cn', 'accuracy'], ['mathbench-primary-cloze_en', 'accuracy']]},
    {'name': 'arithmetic', 'subsets': [['mathbench-arithmetic-cloze_en', 'accuracy']]},
    {'name': 'mathbench-a-cn-average', 'subsets': ['mathbench-college-single_choice_cn', 'mathbench-high-single_choice_cn', 'mathbench-middle-single_choice_cn', 'mathbench-primary-cloze_cn']},
    {'name': 'mathbench-a-en-average', 'subsets': ['mathbench-college-single_choice_en', 'mathbench-high-single_choice_en', 'mathbench-middle-single_choice_en', 'mathbench-primary-cloze_en']},
    {'name': 'mathbench-a (average)', 'subsets': ['college', 'high', 'middle', 'primary', 'arithmetic']},

    {'name': 'college_knowledge', 'subsets': [['mathbench-college_knowledge-single_choice_cn', 'perf_4'], ['mathbench-college_knowledge-single_choice_en', 'perf_4']]},
    {'name': 'high_knowledge', 'subsets': [['mathbench-high_knowledge-single_choice_cn', 'perf_4'], ['mathbench-high_knowledge-single_choice_en', 'perf_4']]},
    {'name': 'middle_knowledge', 'subsets': [['mathbench-middle_knowledge-single_choice_cn', 'perf_4'], ['mathbench-middle_knowledge-single_choice_en', 'perf_4']]},
    {'name': 'primary_knowledge', 'subsets': [['mathbench-primary_knowledge-single_choice_cn', 'perf_4'], ['mathbench-primary_knowledge-single_choice_en', 'perf_4']]},
    {'name': 'mathbench-t-cn-average', 'subsets': ['mathbench-college_knowledge-single_choice_cn', 'mathbench-high_knowledge-single_choice_cn', 'mathbench-middle_knowledge-single_choice_cn', 'mathbench-primary_knowledge-single_choice_cn']},
    {'name': 'mathbench-t-en-average', 'subsets': ['mathbench-college_knowledge-single_choice_en', 'mathbench-high_knowledge-single_choice_en', 'mathbench-middle_knowledge-single_choice_en', 'mathbench-primary_knowledge-single_choice_en']},
    {'name': 'mathbench-t (average)', 'subsets': ['college_knowledge', 'high_knowledge', 'middle_knowledge', 'primary_knowledge']},

    {'name': 'Overall', 'subsets': ['mathbench-a (average)', 'mathbench-t (average)']},
]


summarizer = dict(
    dataset_abbrs = [
        '########################################################',
        '###### MathBench-A-CN: Application Part (Chinese) ######',
        'mathbench-college-single_choice_cn',
        'mathbench-high-single_choice_cn',
        'mathbench-middle-single_choice_cn',
        'mathbench-primary-cloze_cn',
        'mathbench-a-cn-average',

        '###### MathBench-A-EN: Application Part (English) ######',
        'mathbench-college-single_choice_en',
        'mathbench-high-single_choice_en',
        'mathbench-middle-single_choice_en',
        'mathbench-primary-cloze_en',
        'mathbench-a-en-average',

        '###################################################',
        '###### MathBench-T-CN: Theory Part (Chinese) ######',
        'mathbench-college_knowledge-single_choice_cn',
        'mathbench-high_knowledge-single_choice_cn',
        'mathbench-middle_knowledge-single_choice_cn',
        'mathbench-primary_knowledge-single_choice_cn',
        'mathbench-t-cn-average',

        '###### MathBench-T-EN: Theory Part (English) ######',
        'mathbench-college_knowledge-single_choice_en',
        'mathbench-high_knowledge-single_choice_en',
        'mathbench-middle_knowledge-single_choice_en',
        'mathbench-primary_knowledge-single_choice_en',
        'mathbench-t-en-average',
    ],
    summary_groups=mathbench_2024_summary_groups,
)
