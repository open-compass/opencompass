
tasks = ['Code_and_AI', 'Creation', 'LanTask', 'IF', 'chatQA', 'Hallucination', 'safe', 'Reason_and_analysis', 'Longtext', 'Knowledge']
Judgerbenchv2_summary_names = [[task, 'final_score'] for task in tasks]


Judgerbenchv2_summary_groups = [
    {'name': 'Judgerbenchv2', 'subsets': [[name, metric] for name, metric in Judgerbenchv2_summary_names]}
]


summarizer = dict(
    dataset_abbrs=[
        'Judgerbenchv2'
    ],
    summary_groups=Judgerbenchv2_summary_groups,
)