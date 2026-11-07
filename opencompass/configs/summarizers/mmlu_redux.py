from mmengine.config import read_base

with read_base():
    from .groups.mmlu_redux import mmlu_redux_summary_groups

summarizer = dict(
    dataset_abbrs=[
        'mmlu_redux',
        'mmlu_redux-humanities',
        'mmlu_redux-stem',
        'mmlu_redux-social-science',
        'mmlu_redux-other',
    ],
    summary_groups=sum([v for k, v in locals().items() if k.endswith('_summary_groups')], []),
)
