# This summarizer is used for `./datasets/compassbench_v1_math/compassbench_v1_math_gen`

compassbench_v1_math_groups = [
    {'name': 'math_acc_1_and_fill_in_blank', 'subsets': [
        ['compassbench_v1_math-high-single_choice_cn', 'acc_1'],
        ['compassbench_v1_math-high-single_choice_en', 'acc_1'],
        ['compassbench_v1_math-middle-single_choice_cn', 'acc_1'],
        ['compassbench_v1_math-middle-single_choice_en', 'acc_1'],
        ['compassbench_v1_math-primary-cloze_cn', 'accuracy'],
        ['compassbench_v1_math-primary-cloze_en', 'accuracy'],
    ]},
    {'name': 'math_perf_4_and_fill_in_blank', 'subsets': [
        ['compassbench_v1_math-high-single_choice_cn', 'perf_4'],
        ['compassbench_v1_math-high-single_choice_en', 'perf_4'],
        ['compassbench_v1_math-middle-single_choice_cn', 'perf_4'],
        ['compassbench_v1_math-middle-single_choice_en', 'perf_4'],
        ['compassbench_v1_math-primary-cloze_cn', 'accuracy'],
        ['compassbench_v1_math-primary-cloze_en', 'accuracy'],
    ]},
]


summarizer = dict(
    dataset_abbrs=[
        'math_acc_1_and_fill_in_blank',
        ['compassbench_v1_math-high-single_choice_cn', 'acc_1'],
        ['compassbench_v1_math-high-single_choice_en', 'acc_1'],
        ['compassbench_v1_math-middle-single_choice_cn', 'acc_1'],
        ['compassbench_v1_math-middle-single_choice_en', 'acc_1'],
        ['compassbench_v1_math-primary-cloze_cn', 'accuracy'],
        ['compassbench_v1_math-primary-cloze_en', 'accuracy'],

        'math_perf_4_and_fill_in_blank',
        ['compassbench_v1_math-high-single_choice_cn', 'perf_4'],
        ['compassbench_v1_math-high-single_choice_en', 'perf_4'],
        ['compassbench_v1_math-middle-single_choice_cn', 'perf_4'],
        ['compassbench_v1_math-middle-single_choice_en', 'perf_4'],
        ['compassbench_v1_math-primary-cloze_cn', 'accuracy'],
        ['compassbench_v1_math-primary-cloze_en', 'accuracy'],
    ],
    summary_groups=compassbench_v1_math_groups,
)
