# Same scoring as the official RPC-Bench leaderboard
# (https://huggingface.co/spaces/zai-org/RPC-Bench, data/leaderboard.json):
# the overall score is the Informativeness of the open questions, listed
# with Conciseness / Correctness / Completeness / F1-like. All five come
# from rpcbench_test_open, so nothing is averaged across datasets.
# Claim_Verification is not part of the official overall score and is shown
# as a separate group.
_rpcbench_metrics = [
    'informativeness', 'f1_like', 'correctness', 'completeness', 'conciseness'
]

# Groups sharing a name accumulate metrics; the first one (informativeness)
# is what `rpcbench` shows when no metric is given.
rpcbench_summary_groups = [
    dict(name='rpcbench',
         subsets=[['rpcbench_test_open', metric]],
         metric=metric) for metric in _rpcbench_metrics
] + [
    dict(name='rpcbench_claim_verification',
         subsets=[['rpcbench_test_cv', 'accuracy']],
         metric='accuracy')
]
