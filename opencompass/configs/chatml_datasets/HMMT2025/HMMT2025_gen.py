
datasets = [
    dict(
        abbr='HMMT2025',
        path='./data/hmmt2025.jsonl',
        evaluator=dict(
            type='llm_evaluator',
            judge_cfg=dict(),
        ),
        n=1,
    ),
]