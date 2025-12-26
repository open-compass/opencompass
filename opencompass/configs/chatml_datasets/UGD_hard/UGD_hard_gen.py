
datasets = [
    dict(
        abbr='UGD_hard',
        path='./data/UGD_hard_oc.jsonl',
        evaluator=dict(
            type='llm_evaluator',
            judge_cfg=dict(),
        ),
        n=1,
    ),
]