from mmengine.config import read_base

with read_base():
    from .groups.PMMEval import PMMEval_summary_groups


summarizer = dict(
    dataset_abbrs=[
        'flores',
        'humanevalxl',
        'mgsm',
        'mhellaswag',
        'mifeval',
        'mlogiqa',
        'mmmlu',
        'xnli'
    ],
    summary_groups=sum(
        [v for k, v in locals().items() if k.endswith('_summary_groups')], []
    ),
)

