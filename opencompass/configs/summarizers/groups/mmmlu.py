categories = ['mmlu_AR-XY','mmlu_BN-BD','mmlu_DE-DE','mmlu_ES-LA','mmlu_FR-FR','mmlu_HI-IN','mmlu_ID-ID','mmlu_IT-IT','mmlu_JA-JP','mmlu_KO-KR','mmlu_PT-BR','mmlu_SW-KE','mmlu_YO-NG','mmlu_ZH-CN']

mmlu_pro_summary_groups = [
    {'name': 'mmmlu', 'subsets': [c.replace(' ', '_') for c in categories]},
]
