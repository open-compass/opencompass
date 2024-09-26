from mmengine.config import read_base

with read_base():
    from .groups.mmmlu import mmmlu_summary_groups

summarizer = dict(
    dataset_abbrs=[
        'mmlu_AR-XY',
        'mmlu_BN-BD',
        'mmlu_DE-DE',
        'mmlu_ES-LA',
        'mmlu_FR-FR',
        'mmlu_HI-IN',
        'mmlu_ID-ID',
        'mmlu_IT-IT',
        'mmlu_JA-JP',
        'mmlu_KO-KR',
        'mmlu_PT-BR',
        'mmlu_SW-KE',
        'mmlu_YO-NG',
        'mmlu_ZH-CN',
    ],
    summary_groups=sum([v for k, v in locals().items() if k.endswith('_summary_groups')], []),
)
