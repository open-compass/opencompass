
datasets = [
    dict(
        abbr='IMO-Bench-AnswerBench',
        path='./data/imo-bench-answerbench.jsonl',
        evaluator=dict(
            type='llm_evaluator',
            judge_cfg=dict(),
        ),
        n=1,
    ),
]