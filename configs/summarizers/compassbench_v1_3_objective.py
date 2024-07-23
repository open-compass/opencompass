# from mmengine.config import read_base

# with read_base():
#     from .groups.legacy.cibench import cibench_summary_groups
#     from .groups.plugineval import plugineval_summary_groups

compassbench_v13_names = [
    # ["compassbench-knowledge-single_choice_cncircular", "acc_1"],
    ["compassbench-knowledge-single_choice_cncircular", "perf_4"],
    # ["compassbench-math-single_choice_cncircular", "acc_1"],
    ["compassbench-math-single_choice_cncircular", "perf_4"],
]


summarizer = dict(
    dataset_abbrs=[
        ["average", "naive_average"],
        "",
        ["compassbench-knowledge-single_choice_cncircular", "perf_4"],
        ["compassbench-math-single_choice_cncircular", "perf_4"],
    ],
    summary_groups=[
        {
            "name": "average",
            "subsets": [
                ["compassbench-knowledge-single_choice_cncircular", "perf_4"],
                ["compassbench-math-single_choice_cncircular", "perf_4"],
            ],
        }
    ],
)
