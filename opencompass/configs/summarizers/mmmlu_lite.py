categories = ['mmlu_lite_AR-XY','mmlu_lite_BN-BD','mmlu_lite_DE-DE','mmlu_lite_ES-LA','mmlu_lite_FR-FR','mmlu_lite_HI-IN','mmlu_lite_ID-ID','mmlu_lite_IT-IT','mmlu_lite_JA-JP','mmlu_lite_KO-KR','mmlu_lite_PT-BR','mmlu_lite_SW-KE','mmlu_lite_YO-NG','mmlu_lite_ZH-CN']

mmmlu_summary_groups = [
    {'name': 'mmmlu_lite', 'subsets': [f'openai_m{c}' for c in categories]},
]

summarizer = dict(
    dataset_abbrs=[
        'openai_mmmlu_lite_AR-XY',
        'openai_mmmlu_lite_BN-BD',
        'openai_mmmlu_lite_DE-DE',
        'openai_mmmlu_lite_ES-LA',
        'openai_mmmlu_lite_FR-FR',
        'openai_mmmlu_lite_HI-IN',
        'openai_mmmlu_lite_ID-ID',
        'openai_mmmlu_lite_IT-IT',
        'openai_mmmlu_lite_JA-JP',
        'openai_mmmlu_lite_KO-KR',
        'openai_mmmlu_lite_PT-BR',
        'openai_mmmlu_lite_SW-KE',
        'openai_mmmlu_lite_YO-NG',
        'openai_mmmlu_lite_ZH-CN',
        'mmmlu_lite'
    ],
    summary_groups=sum([v for k, v in locals().items() if k.endswith('_summary_groups')], []),
)
