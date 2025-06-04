RewardBench_summary_groups = []

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
RewardBench_summary_groups.append({'name': 'Chat', 'subsets': list(_Chat_weights.keys()), 'weights': _Chat_weights})
RewardBench_summary_groups.append({'name': 'Chat Hard', 'subsets': list(_Chat_Hard_weights.keys()), 'weights': _Chat_Hard_weights})
RewardBench_summary_groups.append({'name': 'Safety', 'subsets': list(_Safety_weights.keys()), 'weights': _Safety_weights})
RewardBench_summary_groups.append({'name': 'Reasoning', 'subsets': list(_Reasoning_weights.keys()), 'weights': _Reasoning_weights})
RewardBench_summary_groups.append({'name': 'RewardBench', 'subsets': list(_RewardBench_weights.keys()), 'weights': _RewardBench_weights})

summarizer = dict(
    dataset_abbrs=[
        'Chat',
        'Chat Hard',
        'Safety',
        'Reasoning',
        'RewardBench'
    ],
    summary_groups=RewardBench_summary_groups,
)