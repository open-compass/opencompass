from mmengine.config import read_base

with read_base():
    from .groups.scicode import scicode_summary_groups

summarizer = dict(
    dataset_abbrs=[
        ['SciCode_with_background', 'accuracy'],
        ['SciCode_with_background', 'sub_accuracy'],
        ['SciCode_wo_background', 'accuracy'],
        ['SciCode_wo_background', 'sub_accuracy'],
    ],
    summary_groups=sum(
        [v for k, v in locals().items() if k.endswith('_summary_groups')], [])
)
