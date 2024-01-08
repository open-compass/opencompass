# This summarizer is used for `./datasets/compassbench_v1_knowledge/compassbench_v1_knowledge_gen`
compassbench_v1_knowledge_names = [
    'compassbench_v1_knowledge-common_knowledge-single_choice_cn_circular',
    'compassbench_v1_knowledge-engineering-single_choice_cn_circular',
    'compassbench_v1_knowledge-humanity-single_choice_cn_circular',
    'compassbench_v1_knowledge-natural_science-single_choice_cn_circular',
    'compassbench_v1_knowledge-social_science-single_choice_cn_circular',
]

compassbench_v1_knowledge_groups = [
    {'name': 'knowledge_cn', 'subsets': compassbench_v1_knowledge_names},
    {'name': 'knowledge_acc_1_and_cloze', 'subsets': [['knowledge_cn', 'acc_1'], ['compassbench_v1_knowledge-mixed-cloze_en', 'score']]},
    {'name': 'knowledge_perf_4_and_cloze', 'subsets': [['knowledge_cn', 'perf_4'], ['compassbench_v1_knowledge-mixed-cloze_en', 'score']]},
]

'compassbench_v1_knowledge-mixed-cloze_en'
summarizer = dict(
    dataset_abbrs=[
        'knowledge_acc_1_and_cloze',
        ['knowledge_cn', 'acc_1'],
        ['compassbench_v1_knowledge-common_knowledge-single_choice_cn_circular', 'acc_1'],
        ['compassbench_v1_knowledge-engineering-single_choice_cn_circular', 'acc_1'],
        ['compassbench_v1_knowledge-humanity-single_choice_cn_circular', 'acc_1'],
        ['compassbench_v1_knowledge-natural_science-single_choice_cn_circular', 'acc_1'],
        ['compassbench_v1_knowledge-social_science-single_choice_cn_circular', 'acc_1'],
        'compassbench_v1_knowledge-mixed-cloze_en',

        'knowledge_perf_4_and_cloze',
        ['knowledge_cn', 'perf_4'],
        ['compassbench_v1_knowledge-common_knowledge-single_choice_cn_circular', 'perf_4'],
        ['compassbench_v1_knowledge-engineering-single_choice_cn_circular', 'perf_4'],
        ['compassbench_v1_knowledge-humanity-single_choice_cn_circular', 'perf_4'],
        ['compassbench_v1_knowledge-natural_science-single_choice_cn_circular', 'perf_4'],
        ['compassbench_v1_knowledge-social_science-single_choice_cn_circular', 'perf_4'],
        'compassbench_v1_knowledge-mixed-cloze_en',
    ],
    summary_groups=compassbench_v1_knowledge_groups
)
