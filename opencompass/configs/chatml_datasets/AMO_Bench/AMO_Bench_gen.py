
datasets = [
    dict(
        abbr='AMO-Bench',
        path='./data/amo-bench.jsonl',
        evaluator=dict(
            type='llm_evaluator',
            judge_cfg=dict(),
        ),
        n=1,
    ),
]