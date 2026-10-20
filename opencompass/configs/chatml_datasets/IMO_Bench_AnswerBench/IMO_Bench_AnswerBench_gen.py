
datasets = [
    dict(
        abbr='IMO-Bench-AnswerBench',
        path='opencompass/IMO-Answer-Bench',
        evaluator=dict(
            type='llm_evaluator',
            judge_cfg=dict(),
        ),
        n=1,
    ),
]