from mmengine.config import read_base

with read_base():
    from .groups.infinitebench import infinitebench_summary_groups

summarizer = dict(
    summary_groups=sum([v for k, v in locals().items() if k.endswith('_summary_groups')], []),
)
