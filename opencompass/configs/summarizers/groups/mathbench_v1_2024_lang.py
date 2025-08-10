# Language specific summarizer groups for MathBench

mathbench_2024_summary_groups = [
    # mathbench-a average with subsets
    {'name': 'college', 'subsets': [['mathbench-college-single_choice_cn', 'perf_4'], ['mathbench-college-single_choice_en', 'perf_4']]},
    {'name': 'high', 'subsets': [['mathbench-high-single_choice_cn', 'perf_4'], ['mathbench-high-single_choice_en', 'perf_4']]},
    {'name': 'middle', 'subsets': [['mathbench-middle-single_choice_cn', 'perf_4'], ['mathbench-middle-single_choice_en', 'perf_4']]},
    {'name': 'primary', 'subsets': [['mathbench-primary-cloze_cn', 'accuracy'], ['mathbench-primary-cloze_en', 'accuracy']]},
    {'name': 'arithmetic', 'subsets': [['mathbench-arithmetic-cloze_en', 'accuracy']]},
    {'name': 'mathbench-a (average)', 'subsets': ['college', 'high', 'middle', 'primary', 'arithmetic']},

    # mathbench-a language
    {'name': 'mathbench-a-college-cn', 'subsets': [['mathbench-college-single_choice_cn', 'perf_4']]},
    {'name': 'mathbench-a-college-en', 'ssubsets': [['mathbench-college-single_choice_en', 'perf_4']]},
    {'name': 'mathbench-a-high-cn', 'subsets': [['mathbench-high-single_choice_cn', 'perf_4']]},
    {'name': 'mathbench-a-high-en', 'subsets': [['mathbench-high-single_choice_en', 'perf_4']]},
    {'name': 'mathbench-a-middle-cn', 'subsets': [['mathbench-middle-single_choice_cn', 'perf_4']]},
    {'name': 'mathbench-a-middle-en', 'subsets': [['mathbench-middle-single_choice_en', 'perf_4']]},
    {'name': 'mathbench-a-primary-cn', 'subsets': [['mathbench-primary-cloze_cn', 'accuracy']]},
    {'name': 'mathbench-a-primary-en', 'subsets': [['mathbench-primary-cloze_en', 'accuracy']]},
    {'name': 'mathbench-a-arithmetic', 'subsets': [['mathbench-arithmetic-cloze_en', 'accuracy']]},
    {'name': 'mathbench-a-cn-average', 'subsets': ['mathbench-a-college-cn', 'mathbench-a-high-cn', 'mathbench-a-middle-cn', 'mathbench-a-primary-cn']},
    {'name': 'mathbench-a-en-average', 'subsets': ['mathbench-a-college-en', 'mathbench-a-high-en', 'mathbench-a-middle-en', 'mathbench-a-primary-en']},
    # mathbench-a average
    {'name': 'mathbench-a (average)', 'subsets': ['mathbench-a-cn-average', 'mathbench-a-en-average']},

    # mathbench-t average with subsets
    {'name': 'college_knowledge', 'subsets': [['mathbench-college_knowledge-single_choice_cn', 'perf_4'], ['mathbench-college_knowledge-single_choice_en', 'perf_4']]},
    {'name': 'high_knowledge', 'subsets': [['mathbench-high_knowledge-single_choice_cn', 'perf_4'], ['mathbench-high_knowledge-single_choice_en', 'perf_4']]},
    {'name': 'middle_knowledge', 'subsets': [['mathbench-middle_knowledge-single_choice_cn', 'perf_4'], ['mathbench-middle_knowledge-single_choice_en', 'perf_4']]},
    {'name': 'college_knowledge', 'subsets': [['mathbench-college_knowledge-single_choice_cn', 'perf_4'], ['mathbench-college_knowledge-single_choice_en', 'perf_4']]},
    {'name': 'high_knowledge', 'subsets': [['mathbench-high_knowledge-single_choice_cn', 'perf_4'], ['mathbench-high_knowledge-single_choice_en', 'perf_4']]},
    {'name': 'middle_knowledge', 'subsets': [['mathbench-middle_knowledge-single_choice_cn', 'perf_4'], ['mathbench-middle_knowledge-single_choice_en', 'perf_4']]},
    {'name': 'primary_knowledge', 'subsets': [['mathbench-primary_knowledge-single_choice_cn', 'perf_4'], ['mathbench-primary_knowledge-single_choice_en', 'perf_4']]},
    # mathbench-t language
    {'name': 'mathbench-t-college-cn', 'subsets': [['mathbench-college_knowledge-single_choice_cn', 'perf_4']]},
    {'name': 'mathbench-t-college-en', 'subsets': [['mathbench-college_knowledge-single_choice_en', 'perf_4']]},
    {'name': 'mathbench-t-high-cn', 'subsets': [['mathbench-high_knowledge-single_choice_cn', 'perf_4']]},
    {'name': 'mathbench-t-high-en', 'subsets': [['mathbench-high_knowledge-single_choice_en', 'perf_4']]},
    {'name': 'mathbench-t-middle-cn', 'subsets': [['mathbench-middle_knowledge-single_choice_cn', 'perf_4']]},
    {'name': 'mathbench-t-middle-en', 'subsets': [['mathbench-middle_knowledge-single_choice_en', 'perf_4']]},
    {'name': 'mathbench-t-primary-cn', 'subsets': [['mathbench-primary_knowledge-single_choice_cn', 'perf_4']]},
    {'name': 'mathbench-t-primary-en', 'subsets': [['mathbench-primary_knowledge-single_choice_en', 'perf_4']]},
    {'name': 'mathbench-t-cn-average', 'subsets': ['mathbench-t-college-cn', 'mathbench-t-high-cn', 'mathbench-t-middle-cn', 'mathbench-t-primary-cn']},
    {'name': 'mathbench-t-en-average', 'subsets': ['mathbench-t-college-en', 'mathbench-t-high-en', 'mathbench-t-middle-en', 'mathbench-t-primary-en']},
    # mathbench-t average
    {'name': 'mathbench-t (average)', 'subsets': ['mathbench-t-cn-average', 'mathbench-t-en-average']},

    # overall cn
    {'name': 'college-cn', 'subsets': ['mathbench-a-college-cn', 'mathbench-t-college-cn']},
    {'name': 'high-cn', 'subsets': ['mathbench-a-high-cn', 'mathbench-t-high-cn']},
    {'name': 'middle-cn', 'subsets': ['mathbench-a-middle-cn', 'mathbench-t-middle-cn']},
    {'name': 'primary-cn', 'subsets': ['mathbench-a-primary-cn', 'mathbench-t-primary-cn']},
    {'name': 'cn-avarage', 'subsets': ['college-cn', 'high-cn', 'middle-cn', 'primary-cn']},

    # overall en
    {'name': 'college-en', 'subsets': ['mathbench-a-college-en', 'mathbench-t-college-en']},
    {'name': 'high-en', 'subsets': ['mathbench-a-high-en', 'mathbench-t-high-en']},
    {'name': 'middle-en', 'subsets': ['mathbench-a-middle-en', 'mathbench-t-middle-en']},
    {'name': 'primary-en', 'subsets': ['mathbench-a-primary-en', 'mathbench-t-primary-en']},
    {'name': 'en-avarage', 'subsets': ['college-en', 'high-en', 'middle-en', 'primary-en']},

    # overall
    {'name': 'Overall', 'subsets': ['mathbench-a (average)', 'mathbench-t (average)']},
]


summarizer = dict(
    dataset_abbrs = [
        '########################################################',
        '###### MathBench-A-CN: Application Part (Chinese) ######',
        'mathbench-a-college-cn',
        'mathbench-a-high-cn',
        'mathbench-a-middle-cn',
        'mathbench-a-primary-cn',
        'mathbench-a-cn-average',
        '###### MathBench-A-EN: Application Part (English) ######',
        'mathbench-a-college-en',
        'mathbench-a-high-en',
        'mathbench-a-middle-en',
        'mathbench-a-primary-en',
        'mathbench-a-en-average',
        '#########################################################',
        '###### MathBench-T-CN: Theory Part (Chinese) ############',
        'mathbench-t-college-cn',
        'mathbench-t-high-cn',
        'mathbench-t-middle-cn',
        'mathbench-t-primary-cn',
        'mathbench-t-cn-average',
        '###### MathBench-T-EN: Theory Part (English) ############',
        'mathbench-t-college-en',
        'mathbench-t-high-en',
        'mathbench-t-middle-en',
        'mathbench-t-primary-en',
        'mathbench-t-en-average',
        '#########################################################',
        '###### MathBench-CN ############',
        'college-cn',
        'high-cn',
        'middle-cn',
        'primary-cn',
        'cn-avarage',
        '###### MathBench-EN ############',
        'college-en',
        'high-en',
        'middle-en',
        'primary-en',
        'en-avarage',
        '#########################################################',
    ],
    summary_groups=mathbench_2024_summary_groups,
)
