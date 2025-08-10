from mmengine.config import read_base

with read_base():
    from .groups.mmmlu import mmmlu_summary_groups

summarizer = dict(
    dataset_abbrs=[
        'openai_mmmlu_AR-XY',
        'openai_mmmlu_BN-BD',
        'openai_mmmlu_DE-DE',
        'openai_mmmlu_ES-LA',
        'openai_mmmlu_FR-FR',
        'openai_mmmlu_HI-IN',
        'openai_mmmlu_ID-ID',
        'openai_mmmlu_IT-IT',
        'openai_mmmlu_JA-JP',
        'openai_mmmlu_KO-KR',
        'openai_mmmlu_PT-BR',
        'openai_mmmlu_SW-KE',
        'openai_mmmlu_YO-NG',
        'openai_mmmlu_ZH-CN',
        'mmmlu',
    ],
    summary_groups=sum([v for k, v in locals().items() if k.endswith('_summary_groups')], []),
)
