from mmengine.config import read_base

with read_base():
    from .groups.mmlu_cf import mmlu_cf_summary_groups

summarizer = dict(
    dataset_abbrs=[
        'mmlu_cf_Biology',
        'mmlu_cf_Business',
        'mmlu_cf_Chemistry',
        'mmlu_cf_Computer_Science',
        'mmlu_cf_Economics',
        'mmlu_cf_Engineering',
        'mmlu_cf_Health',
        'mmlu_cf_History',
        'mmlu_cf_Law',
        'mmlu_cf_Math',
        'mmlu_cf_Philosophy',
        'mmlu_cf_Physics',
        'mmlu_cf_Psychology',
        'mmlu_cf_Other',
        'mmlu_cf',
    ],
    summary_groups=sum([v for k, v in locals().items() if k.endswith('_summary_groups')], []),
)
