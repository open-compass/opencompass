from mmengine.config import read_base

with read_base():
    from .groups.agieval import agieval_summary_groups
    from .groups.mmlu import mmlu_summary_groups
    from .groups.cmmlu import cmmlu_summary_groups
    from .groups.ceval import ceval_summary_groups
    from .groups.bbh import bbh_summary_groups
    from .groups.GaokaoBench import GaokaoBench_summary_groups
    from .groups.flores import flores_summary_groups
    from .groups.tydiqa import tydiqa_summary_groups
    from .groups.xiezhi import xiezhi_summary_groups
    from .groups.scibench import scibench_summary_groups
    from .groups.mgsm import mgsm_summary_groups
    from .groups.longbench import longbench_summary_groups

summarizer = dict(
    summary_groups=sum([v for k, v in locals().items() if k.endswith('_summary_groups')], []),
)
