from mmengine.config import read_base

with read_base():
    from .groups.bbeh import bbeh_summary_groups

summarizer = dict(
    dataset_abbrs=[
        ['bbeh', 'naive_average'],
        ['bbeh', 'harmonic_mean']
    ],
    summary_groups=sum(
        [v for k, v in locals().items() if k.endswith('_summary_groups')], []),
)