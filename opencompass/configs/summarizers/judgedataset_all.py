Judge_all_summary_groups = []


# RewardBench
_Chat_weights = {
'alpacaeval-easy': 0.32355305466237944,
'alpacaeval-length': 0.32355305466237944,
'alpacaeval-hard': 0.32355305466237944,
'mt-bench-easy': 0.011254019292604502,
'mt-bench-med': 0.018086816720257234,
}

_Chat_Hard_weights = {
'mt-bench-hard': 0.09698275862068965,
'llmbar-natural': 0.21551724137931033,
'llmbar-adver-neighbor': 0.28879310344827586,
'llmbar-adver-GPTInst': 0.19827586206896552,
'llmbar-adver-GPTOut': 0.10129310344827586,
'llmbar-adver-manual': 0.09913793103448276,
}

_Safety_weights = {
'refusals-dangerous': 0.13513513513513514,
'refusals-offensive': 0.13513513513513514,
'xstest-should-refuse': 0.20810810810810812,
'xstest-should-respond': 0.33783783783783783,
'donotanswer': 0.1837837837837838,
}

_Reasoning_weights = {
'math-prm': 0.31236897274633124,
'hep-cpp': 0.1146051712089448,
'hep-go': 0.1146051712089448,
'hep-java': 0.1146051712089448,
'hep-js': 0.1146051712089448,
'hep-python': 0.1146051712089448,
'hep-rust': 0.1146051712089448,
}

_RewardBench_weights = {'alpacaeval-easy': 0.08088826366559486,'alpacaeval-length': 0.08088826366559486,'alpacaeval-hard': 0.08088826366559486,'mt-bench-easy': 0.0028135048231511255,'mt-bench-med': 0.004521704180064309,'mt-bench-hard': 0.024245689655172414,'llmbar-natural': 0.05387931034482758,'llmbar-adver-neighbor': 0.07219827586206896,'llmbar-adver-GPTInst': 0.04956896551724138,'llmbar-adver-GPTOut': 0.025323275862068964,'llmbar-adver-manual': 0.02478448275862069,'refusals-dangerous': 0.033783783783783786,'refusals-offensive': 0.033783783783783786,'xstest-should-refuse': 0.05202702702702703,'xstest-should-respond': 0.08445945945945946,'donotanswer': 0.04594594594594595,'math-prm': 0.07809224318658281,'hep-cpp': 0.0286512928022362,'hep-go': 0.0286512928022362,'hep-java': 0.0286512928022362,'hep-js': 0.0286512928022362,'hep-python': 0.0286512928022362,'hep-rust': 0.0286512928022362,}

Judge_all_summary_groups.append({'name': 'RewardBench_avg', 'subsets': list(_RewardBench_weights.keys()), 'weights': _RewardBench_weights})
Judge_all_summary_groups.append({'name': 'RewardBench_Chat', 'subsets': list(_Chat_weights.keys()), 'weights': _Chat_weights})
Judge_all_summary_groups.append({'name': 'RewardBench_Chat Hard', 'subsets': list(_Chat_Hard_weights.keys()), 'weights': _Chat_Hard_weights})
Judge_all_summary_groups.append({'name': 'RewardBench_Safety', 'subsets': list(_Safety_weights.keys()), 'weights': _Safety_weights})
Judge_all_summary_groups.append({'name': 'RewardBench_Reasoning', 'subsets': list(_Reasoning_weights.keys()), 'weights': _Reasoning_weights})



# Judgerbenchv2
Judgerbenchv2_tasks = ['Code_and_AI', 'Creation', 'LanTask', 'IF', 'chatQA', 'Hallucination', 'safe', 'Reason_and_analysis', 'Longtext', 'Knowledge']
Judgerbenchv2_metrics = ['final_score', 'accuracy', 'normalized_diff', 'rank_diff', 'score_diff']
Judgerbenchv2_summary_names = []
for metric in Judgerbenchv2_metrics:
    for task in Judgerbenchv2_tasks:
        Judgerbenchv2_summary_names.append([task, metric])

Judge_all_summary_groups.append({'name': 'Judgerbenchv2_final_score', 'subsets': [[name, metric] for name, metric in Judgerbenchv2_summary_names if metric == 'final_score']})
Judge_all_summary_groups.append({'name': 'Judgerbenchv2_accuracy', 'subsets': [[name, metric] for name, metric in Judgerbenchv2_summary_names if metric == 'accuracy']})
Judge_all_summary_groups.append({'name': 'Judgerbenchv2_normalized_diff', 'subsets': [[name, metric] for name, metric in Judgerbenchv2_summary_names if metric == 'normalized_diff']})
Judge_all_summary_groups.append({'name': 'Judgerbenchv2_rank_diff', 'subsets': [[name, metric] for name, metric in Judgerbenchv2_summary_names if metric == 'rank_diff']})
Judge_all_summary_groups.append({'name': 'Judgerbenchv2_score_diff', 'subsets': [[name, metric] for name, metric in Judgerbenchv2_summary_names if metric == 'score_diff']})

Judge_all_summary_groups.append({'name': 'Judgebench', 'subsets': ['judgebench']})
Judge_all_summary_groups.append({'name': 'rmb_dataset_total_avg', 'subsets': [['rmb_dataset', 'total_accuracy']]})
Judge_all_summary_groups.append({'name': 'rmb_dataset_pair', 'subsets': [['rmb_dataset', 'pair_average']]})
Judge_all_summary_groups.append({'name': 'rmb_dataset_bon', 'subsets': [['rmb_dataset', 'bon_average']]})

summarizer = dict(
    dataset_abbrs=[        
        'Judgerbenchv2_final_score',
        'Judgebench',
        'rmb_dataset_total_avg',
        'RewardBench_avg',
        '',
        'Judgerbenchv2_accuracy',
        'Judgerbenchv2_normalized_diff',
        'Judgerbenchv2_rank_diff',
        'Judgerbenchv2_score_diff',
        '', 
        'rmb_dataset_pair',
        'rmb_dataset_bon',
        '',
        'RewardBench_Chat',
        'RewardBench_Chat Hard',
        'RewardBench_Safety',
        'RewardBench_Reasoning',
    ],
    summary_groups=Judge_all_summary_groups,
)
