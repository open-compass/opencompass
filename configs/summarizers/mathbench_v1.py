summarizer = dict(
    dataset_abbrs=[
        '######## MathBench Accuracy ########', # category
        ['mathbench-college-single_choice_cn', 'acc_1'],
        ['mathbench-college-single_choice_en', 'acc_1'],
        ['mathbench-high-single_choice_cn', 'acc_1'],
        ['mathbench-high-single_choice_en', 'acc_1'],
        ['mathbench-middle-single_choice_cn', 'acc_1'],
        ['mathbench-middle-single_choice_en', 'acc_1'],
        ['mathbench-primary-cloze_cn', 'accuracy'],
        ['mathbench-primary-cloze_en', 'accuracy'],
        ['mathbench-calculate-cloze_en', 'accuracy'],
        '######## MathBench CircularEval ########', # category
        ['mathbench-college-single_choice_cn', 'perf_4'],
        ['mathbench-college-single_choice_en', 'perf_4'],
        ['mathbench-high-single_choice_cn', 'perf_4'],
        ['mathbench-high-single_choice_en', 'perf_4'],
        ['mathbench-middle-single_choice_cn', 'perf_4'],
        ['mathbench-middle-single_choice_en', 'perf_4'],
        '######## MathBench Knowledge ########', # category
        ['mathbench-college_knowledge-single_choice_cn', 'perf_4'],
        ['mathbench-college_knowledge-single_choice_en', 'perf_4'],
        ['mathbench-high_knowledge-single_choice_cn', 'perf_4'],
        ['mathbench-high_knowledge-single_choice_en', 'perf_4'],
        ['mathbench-middle_knowledge-single_choice_cn', 'perf_4'],
        ['mathbench-middle_knowledge-single_choice_en', 'perf_4'],
        ['mathbench-primary_knowledge-single_choice_cn', 'perf_4'],
        ['mathbench-primary_knowledge-single_choice_en', 'perf_4'],
    ],
    summary_groups=sum(
        [v for k, v in locals().items() if k.endswith("_summary_groups")], [])
)
