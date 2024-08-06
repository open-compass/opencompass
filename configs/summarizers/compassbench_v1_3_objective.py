from mmengine.config import read_base

with read_base():
    from .groups.legacy.cibench import cibench_summary_groups
    from .groups.plugineval import plugineval_summary_groups

obj_summary_groups = [
    ########################## knowledge ##########################
    {
        'name': 'knowledge_en',
        'subsets': [
            ['compassbench-wiki_en_sub_500_人文科学_circular', 'perf_4'],
            ['compassbench-wiki_en_sub_500_社会科学_circular', 'perf_4'],
            ['compassbench-wiki_en_sub_500_自然科学-工科_circular', 'perf_4'],
            ['compassbench-wiki_en_sub_500_自然科学-理科_circular', 'perf_4'],
        ],
    },
    {
        'name': 'knowledge_cn',
        'subsets': [
            ['compassbench-wiki_zh_sub_500_人文科学_circular', 'perf_4'],
            ['compassbench-wiki_zh_sub_500_社会科学_circular', 'perf_4'],
            ['compassbench-wiki_zh_sub_500_自然科学-工科_circular', 'perf_4'],
            ['compassbench-wiki_zh_sub_500_自然科学-理科_circular', 'perf_4'],
        ],
    },
    {
        'name': 'knowledge',
        'subsets': [
            ['compassbench-wiki_en_sub_500_人文科学_circular', 'perf_4'],
            ['compassbench-wiki_en_sub_500_社会科学_circular', 'perf_4'],
            ['compassbench-wiki_en_sub_500_自然科学-工科_circular', 'perf_4'],
            ['compassbench-wiki_en_sub_500_自然科学-理科_circular', 'perf_4'],
            ['compassbench-wiki_zh_sub_500_人文科学_circular', 'perf_4'],
            ['compassbench-wiki_zh_sub_500_社会科学_circular', 'perf_4'],
            ['compassbench-wiki_zh_sub_500_自然科学-工科_circular', 'perf_4'],
            ['compassbench-wiki_zh_sub_500_自然科学-理科_circular', 'perf_4'],
        ],
    },
    ########################## math ##########################
    {
        'name': 'math_en',
        'subsets': [
            ['compassbench-college_single_choice_en_circular', 'perf_4'],
            ['compassbench-arithmetic_cloze_en', 'accuracy'],
        ],
    },
    {
        'name': 'math_cn',
        'subsets': [
            ['compassbench-college_single_choice_cn_circular', 'perf_4'],
            ['compassbench-arithmetic_cloze_en', 'accuracy'],
        ],
    },
    {
        'name': 'math',
        'subsets': [
            ['compassbench-college_single_choice_cn_circular', 'perf_4'],
            ['compassbench-college_single_choice_en_circular', 'perf_4'],
            ['compassbench-arithmetic_cloze_en', 'accuracy'],
        ],
    },
    ########################## code ##########################
    {
        'name': 'code-completion_en',
        'subsets': [
            ['compass_bench_cdoe_completion_en', 'humaneval_plus_pass@1'],
        ],
    },
    {
        'name': 'code-completion_cn',
        'subsets': [
            ['compass_bench_cdoe_completion_zh', 'humaneval_pass@1'],
        ],
    },
    {
        'name': 'code-interview_en',
        'subsets': [
            ['compass_bench_code_interview_en-EASY', 'pass@1'],
            ['compass_bench_code_interview_en-MEDIUM', 'pass@1'],
            ['compass_bench_code_interview_en-HARD', 'pass@1'],
        ],
    },
    {
        'name': 'code-interview_cn',
        'subsets': [
            ['compass_bench_code_interview_zh-EASY', 'pass@1'],
            ['compass_bench_code_interview_zh-MEDIUM', 'pass@1'],
            ['compass_bench_code_interview_zh-HARD', 'pass@1'],
        ],
    },
    {
        'name': 'code-competition',
        'subsets': [
            ['TACO-EASY', 'pass@1'],
            ['TACO-MEDIUM', 'pass@1'],
            ['TACO-MEDIUM_HARD', 'pass@1'],
            ['TACO-HARD', 'pass@1'],
            ['TACO-VERY_HARD', 'pass@1'],
        ],
    },
    {
        'name': 'code_cn',
        'subsets': [
            ['code-completion_cn', 'naive_average'],
            ['code-interview_cn', 'naive_average'],
        ],
    },
    {
        'name': 'code_en',
        'subsets': [
            ['code-completion_en', 'naive_average'],
            ['code-interview_en', 'naive_average'],
            ['code-competition', 'naive_average'],
        ],
    },
    {
        'name': 'code',
        'subsets': [
            ['code-completion_cn', 'naive_average'],
            ['code-interview_cn', 'naive_average'],
            ['code-completion_en', 'naive_average'],
            ['code-interview_en', 'naive_average'],
            ['code-competition', 'naive_average'],
        ],
    },
]
agent_summary_groups = [
    dict(
        name='cibench_template',
        subsets=[
            'cibench_template_wo_nltk:executable',
            'cibench_template_wo_nltk:numeric_correct',
            'cibench_template_wo_nltk:vis_sim',
        ],
    ),
    dict(
        name='cibench_template_cn',
        subsets=[
            'cibench_template_cn_wo_nltk:executable',
            'cibench_template_cn_wo_nltk:numeric_correct',
            'cibench_template_cn_wo_nltk:vis_sim',
        ],
    ),
    # dict(
    #     name='agent_cn',
    #     subsets=['cibench_template_cn', 'plugin_eval-mus-p10_one_review_zh'],
    # ),
    # dict(
    #     name='agent_en', subsets=['cibench_template', 'plugin_eval-mus-p10_one_review']
    # ),
        dict(
        name='agent_cn',
        subsets=['plugin_eval-mus-p10_one_review_zh'],
    ),
    dict(
        name='agent_en', subsets=['plugin_eval-mus-p10_one_review']
    ),
    dict(name='agent', subsets=['agent_cn', 'agent_en']),
]


summarizer = dict(
    dataset_abbrs=[
        # ["average", "naive_average"],
        # "",
        ['knowledge', 'naive_average'],
        ['knowledge_en','naive_average'],
        ['knowledge_cn','naive_average'],
        ['compassbench-wiki_en_sub_500_人文科学_circular', 'perf_4'],
        ['compassbench-wiki_en_sub_500_社会科学_circular', 'perf_4'],
        ['compassbench-wiki_en_sub_500_自然科学-工科_circular', 'perf_4'],
        ['compassbench-wiki_en_sub_500_自然科学-理科_circular', 'perf_4'],
        ['compassbench-wiki_zh_sub_500_人文科学_circular', 'perf_4'],
        ['compassbench-wiki_zh_sub_500_社会科学_circular', 'perf_4'],
        ['compassbench-wiki_zh_sub_500_自然科学-工科_circular', 'perf_4'],
        ['compassbench-wiki_zh_sub_500_自然科学-理科_circular', 'perf_4'],
        '',
        ['math', 'naive_average'],
        ['math_en', 'naive_average'],
        ['math_cn', 'naive_average'],
        ['compassbench-college_single_choice_cn_circular', 'perf_4'],
        ['compassbench-college_single_choice_en_circular', 'perf_4'],
        ['compassbench-arithmetic_cloze_en', 'accuracy'],
        '',
        ['code', 'naive_average'],
        ['code_cn', 'naive_average'],
        ['code_en', 'naive_average'],
        ['code-completion_cn', 'naive_average'],
        ['code-completion_en', 'naive_average'],
        ['code-interview_cn', 'naive_average'],
        ['code-interview_en', 'naive_average'],
        ['code-competition', 'naive_average'],
        '',
        ['agent', 'naive_average'],
        ['agent_en', 'naive_average'],
        ['agent_cn', 'naive_average'],
        ['plugin_eval-mus-p10_one_review_zh', 'naive_average'],
        ['plugin_eval-mus-p10_one_review', 'naive_average'],
    ],
    summary_groups=sum(
        [v for k, v in locals().items() if k.endswith('_summary_groups')], []
    ),
)
