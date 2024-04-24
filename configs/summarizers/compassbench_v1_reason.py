compassbench_v1_reason_groups = [
    {'name': 'reasonbench_cn_abductive_circular', 'subsets': ['reasonbench_cn_abductive_alphanlg_translated_circular']},
    {'name': 'reasonbench_en_abductive_circular', 'subsets': ['reasonbench_en_abductive_alphanlg_circular']},
    {'name': 'reasonbench_cn_deductive_circular', 'subsets': ['reasonbench_cn_deductive_bbh3obj_translated_circular', 'reasonbench_cn_deductive_logiqa_zh_circular']},
    {'name': 'reasonbench_cn_inductive_circular', 'subsets': ['reasonbench_cn_inductive_deer_translated_circular', 'reasonbench_cn_inductive_selfgenerated_circular']},
    {'name': 'reasonbench_en_inductive_circular', 'subsets': ['reasonbench_en_inductive_deer_circular', 'reasonbench_en_inductive_selfgenerated_circular']},

    {'name': 'reasonbench_cn_circular', 'subsets': ['reasonbench_cn_commonsense_circular', 'reasonbench_cn_abductive_circular', 'reasonbench_cn_deductive_circular', 'reasonbench_cn_inductive_circular']},
    {'name': 'reasonbench_en_circular', 'subsets': ['reasonbench_en_commonsense_circular', 'reasonbench_en_abductive_circular', 'reasonbench_en_deductive_logiqa_zh_translated_circular', 'reasonbench_en_inductive_circular']},
    {'name': 'reasonbench', 'subsets': ['reasonbench_cn_circular', 'reasonbench_en_circular']},
]

summarizer = dict(
    dataset_abbrs=[
        ['reasonbench', 'perf_circular'],
        ['reasonbench_cn_circular', 'perf_circular'],
        ['reasonbench_en_circular', 'perf_circular'],
        ['reasonbench_cn_commonsense_circular', 'perf_circular'],
        ['reasonbench_cn_abductive_circular', 'perf_circular'],
        ['reasonbench_cn_deductive_circular', 'perf_circular'],
        ['reasonbench_cn_inductive_circular', 'perf_circular'],
        ['reasonbench_en_commonsense_circular', 'perf_circular'],
        ['reasonbench_en_abductive_circular', 'perf_circular'],
        ['reasonbench_en_deductive_logiqa_zh_translated_circular', 'perf_circular'],
        ['reasonbench_en_inductive_circular', 'perf_circular'],
    ],
    summary_groups=compassbench_v1_reason_groups,
)
