
datasets = [
    dict(
        abbr='C-MHChem',
        path='./data/C-MHChem2.jsonl',
        evaluator=dict(
            type='llm_evaluator',
            judge_cfg=dict(),
        ),
        n=1,
    ),
]