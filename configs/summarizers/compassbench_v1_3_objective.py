from mmengine.config import read_base

with read_base():
    from .groups.legacy.cibench import cibench_summary_groups
    from .groups.plugineval import plugineval_summary_groups

# compassbench_v13_names = [
#     ["compassbench-college_single_choice_cn_circular", "perf_4"],
#     ["compassbench-college_single_choice_en_circular", "perf_4"],
#     ["compassbench-arithmetic_cloze_en", "accuracy"],
# ]

obj_summary_groups = [
    {
        "name": "compassbench-knowledge",
        "subsets": [
            ["compassbench-wiki_en_sub_500_人文科学_circular", "perf_4"],
            ["compassbench-wiki_en_sub_500_社会科学_circular", "perf_4"],
            ["compassbench-wiki_en_sub_500_生活常识_circular", "perf_4"],
            ["compassbench-wiki_en_sub_500_自然科学-工科_circular", "perf_4"],
            ["compassbench-wiki_en_sub_500_自然科学-理科_circular", "perf_4"],
            ["compassbench-wiki_zh_sub_500_人文科学_circular", "perf_4"],
            ["compassbench-wiki_zh_sub_500_社会科学_circular", "perf_4"],
            ["compassbench-wiki_zh_sub_500_生活常识_circular", "perf_4"],
            ["compassbench-wiki_zh_sub_500_自然科学-工科_circular", "perf_4"],
            ["compassbench-wiki_zh_sub_500_自然科学-理科_circular", "perf_4"],
        ],
    },
    {
        "name": "compassbench-math",
        "subsets": [
            ["compassbench-college_single_choice_cn_circular", "perf_4"],
            ["compassbench-college_single_choice_en_circular", "perf_4"],
            ["compassbench-arithmetic_cloze_en", "accuracy"],
        ],
    },
    # Code
    {
        "name": "code-completion",
        "subsets": [
            ["compass_bench_cdoe_completion_en", "humaneval_plus_pass@1"],
            ["compass_bench_cdoe_completion_zh", "humaneval_pass@1"],
        ],
    },
    {
        "name": "code-interview",
        "subsets": [
            ["compass_bench_code_interview_en-EASY", "pass@1"],
            ["compass_bench_code_interview_en-MEDIUM", "pass@1"],
            ["compass_bench_code_interview_en-HARD", "pass@1"],
            ["compass_bench_code_interview_zh-EASY", "pass@1"],
            ["compass_bench_code_interview_zh-MEDIUM", "pass@1"],
            ["compass_bench_code_interview_zh-HARD", "pass@1"],
        ],
    },
    {
        "name": "code-competition",
        "subsets": [
            ["TACO-EASY", "pass@1"],
            ["TACO-MEDIUM", "pass@1"],
            ["TACO-MEDIUM_HARD", "pass@1"],
            ["TACO-HARD", "pass@1"],
            ["TACO-VERY_HARD", "pass@1"],
        ],
    },
]
agent_summary_groups = [
    dict(
        name="cibench_template",
        subsets=[
            "cibench_template_wo_nltk:executable",
            "cibench_template_wo_nltk:numeric_correct",
            "cibench_template_wo_nltk:vis_sim",
        ],
    ),
    dict(
        name="cibench_template_cn",
        subsets=[
            "cibench_template_cn_wo_nltk:executable",
            "cibench_template_cn_wo_nltk:numeric_correct",
            "cibench_template_cn_wo_nltk:vis_sim",
        ],
    ),
    dict(
        name="agent_cn",
        subsets=["cibench_template_cn", "plugin_eval-mus-p10_one_review_zh"],
    ),
    dict(
        name="agent_en", subsets=["cibench_template", "plugin_eval-mus-p10_one_review"]
    ),
    dict(name="agent", subsets=["agent_cn", "agent_en"]),
]


summarizer = dict(
    dataset_abbrs=[
        # ["average", "naive_average"],
        # "",
        ["compassbench-knowledge", "naive_average"],
        ["compassbench-wiki_en_sub_500_人文科学_circular", "perf_4"],
        ["compassbench-wiki_en_sub_500_社会科学_circular", "perf_4"],
        ["compassbench-wiki_en_sub_500_生活常识_circular", "perf_4"],
        ["compassbench-wiki_en_sub_500_自然科学-工科_circular", "perf_4"],
        ["compassbench-wiki_en_sub_500_自然科学-理科_circular", "perf_4"],
        ["compassbench-wiki_zh_sub_500_人文科学_circular", "perf_4"],
        ["compassbench-wiki_zh_sub_500_社会科学_circular", "perf_4"],
        ["compassbench-wiki_zh_sub_500_生活常识_circular", "perf_4"],
        ["compassbench-wiki_zh_sub_500_自然科学-工科_circular", "perf_4"],
        ["compassbench-wiki_zh_sub_500_自然科学-理科_circular", "perf_4"],
        "",
        ["compassbench-math", "naive_average"],
        ["compassbench-college_single_choice_cn_circular", "perf_4"],
        ["compassbench-college_single_choice_en_circular", "perf_4"],
        ["compassbench-arithmetic_cloze_en", "accuracy"],
        "",
        ["code-interview", "naive_average"],
        ["code-competition", "naive_average"],
        ["code-completion", "naive_average"],
        "",
        ["agent", "naive_average"],
        ["agent_en", "naive_average"],
        ["agent_cn", "naive_average"],
        # ["cibench_template", "naive_average"],
        # ["cibench_template_cn", "naive_average"],
        # 'plugin_eval-mus-p10_one_review_zh',
        # 'plugin_eval-mus-p10_one_review',
    ],
    summary_groups=sum(
        [v for k, v in locals().items() if k.endswith("_summary_groups")], []
    ),
)
