RewardBench_summary_groups = []

_RewardBench_weights = {'alpacaeval-easy': 0.08088826366559486,'alpacaeval-length': 0.08088826366559486,'alpacaeval-hard': 0.08088826366559486,'mt-bench-easy': 0.0028135048231511255,'mt-bench-med': 0.004521704180064309,'mt-bench-hard': 0.024245689655172414,'llmbar-natural': 0.05387931034482758,'llmbar-adver-neighbor': 0.07219827586206896,'llmbar-adver-GPTInst': 0.04956896551724138,'llmbar-adver-GPTOut': 0.025323275862068964,'llmbar-adver-manual': 0.02478448275862069,'refusals-dangerous': 0.033783783783783786,'refusals-offensive': 0.033783783783783786,'xstest-should-refuse': 0.05202702702702703,'xstest-should-respond': 0.08445945945945946,'donotanswer': 0.04594594594594595,'math-prm': 0.07809224318658281,'hep-cpp': 0.0286512928022362,'hep-go': 0.0286512928022362,'hep-java': 0.0286512928022362,'hep-js': 0.0286512928022362,'hep-python': 0.0286512928022362,'hep-rust': 0.0286512928022362,}
RewardBench_summary_groups.append({'name': 'RewardBench', 'subsets': list(_RewardBench_weights.keys()), 'weights': _RewardBench_weights})

summarizer = dict(
    dataset_abbrs=[
        'RewardBench'
    ],
    summary_groups=RewardBench_summary_groups,
)