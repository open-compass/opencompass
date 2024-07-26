from mmengine.config import read_base

with read_base():
    from .groups.legacy.cibench import cibench_summary_groups
    from .groups.plugineval import plugineval_summary_groups

compassbench_v13_names = [
    ["compassbench-knowledge_circular", "perf_4"],
    ["compassbench-college_single_choice_cn_circular", "perf_4"],
    ["compassbench-college_single_choice_en_circular", "perf_4"],
    ["compassbench-arithmetic_cloze_en", "accuracy"],
]

obj_summary_groups = [
    {
        "name": "compassbench-knowledge",
        "subsets": [["compassbench-knowledge_circular", "perf_4"]],
    },
    {
        "name": "compassbench-math",
        "subsets": [
            ["compassbench-college_single_choice_cn_circular", "perf_4"],
            ["compassbench-college_single_choice_en_circular", "perf_4"],
            ["compassbench-arithmetic_cloze_en", "accuracy"],
        ],
    },
    {
        "name": "compassbench-college",
        "subsets": [
            ["compassbench-college_single_choice_cn_circular", "perf_4"],
            ["compassbench-college_single_choice_en_circular", "perf_4"],
        ],
    },
    # {
    #     "name": "average",
    #     "subsets": [
    #         ["compassbench-knowledge-single_choice_cncircular", "perf_4"],
    #         ["compassbench-math-single_choice_cncircular", "perf_4"],
    #     ],
    # }
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
        ["wiki_en_sub_500_人文科学", "perf_4"],
        ["wiki_en_sub_500_社会科学", "perf_4"],
        ["wiki_en_sub_500_生活常识", "perf_4"],
        ["wiki_en_sub_500_自然科学-工科", "perf_4"],
        ["wiki_en_sub_500_自然科学-理科", "perf_4"],
        ["wiki_zh_sub_500_人文科学", "perf_4"],
        ["wiki_zh_sub_500_社会科学", "perf_4"],
        ["wiki_zh_sub_500_生活常识", "perf_4"],
        ["wiki_zh_sub_500_自然科学-工科", "perf_4"],
        ["wiki_zh_sub_500_自然科学-理科", "perf_4"],
        "",
        ["compassbench-math", "naive_average"],
        ["compassbench-college", "naive_average"],
        ["compassbench-arithmetic_cloze_en", "accuracy"],
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
