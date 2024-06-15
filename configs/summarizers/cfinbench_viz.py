from mmengine.config import read_base

with read_base():
    from .groups.cfinbench import cfinbench_summary_groups

summarizer = dict(
    dataset_abbrs = [
        '---------val zero-shot ---------',
        "subject_weighted_zero_shot",
        "qualification_weighted_zero_shot",
        "practice_weighted_zero_shot",
        "law_weighted_zero_shot",
        '---------val few-shot ---------',
        "subject_weighted_few_shot",
        "qualification_weighted_few_shot",
        "practice_weighted_few_shot",
        "law_weighted_few_shot"
    ],
    summary_groups=sum([v for k, v in locals().items() if k.endswith("_summary_groups")], []),
)
