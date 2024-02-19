summarizer = dict(
    dataset_abbrs=[
        '######## MathBench Accuracy ########', # category
        ['mathbench-college-single_choice_cn', 'acc_1'],
        ['mathbench-college-cloze_en', 'accuracy'],
        ['mathbench-high-single_choice_cn', 'acc_1'],
        ['mathbench-high-single_choice_en', 'acc_1'],
        ['mathbench-middle-single_choice_cn', 'acc_1'],
        ['mathbench-primary-cloze_cn', 'accuracy'],
        '######## MathBench CircularEval ########', # category
        ['mathbench-college-single_choice_cn', 'perf_4'],
        ['mathbench-high-single_choice_cn', 'perf_4'],
        ['mathbench-high-single_choice_en', 'perf_4'],
        ['mathbench-middle-single_choice_cn', 'perf_4'],
    ],
    summary_groups=sum(
        [v for k, v in locals().items() if k.endswith('_summary_groups')], [])
)
