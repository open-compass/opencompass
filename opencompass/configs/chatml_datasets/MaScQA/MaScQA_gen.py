
datasets = [
    dict(
        abbr='MaScQA',
        path='./data/MaScQA/MaScQA.jsonl',
        evaluator=dict(
            type='llm_evaluator',
            judge_cfg=dict(),
        ),
        n=1,
    ),
]