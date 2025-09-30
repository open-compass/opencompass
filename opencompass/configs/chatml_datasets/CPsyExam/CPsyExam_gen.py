
datasets = [
    dict(
        abbr='CPsyExam',
        path='./data/merged_train_dev.jsonl',
        evaluator=dict(
            type='llm_evaluator',
            judge_cfg=dict(),
        ),
        n=1,
    ),
]