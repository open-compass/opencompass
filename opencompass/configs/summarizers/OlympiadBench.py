from mmengine.config import read_base

with read_base():
    from .groups.OlympiadBench import OlympiadBench_summary_groups

summarizer = dict(
    dataset_abbrs=[
        'OlympiadBench_OE_TO_maths_en_COMP',
        'OlympiadBench_OE_TO_maths_zh_COMP',
        'OlympiadBench_OE_TO_maths_zh_CEE',
        'OlympiadBench_OE_TO_physics_en_COMP',
        'OlympiadBench_OE_TO_physics_zh_CEE'
    ],
    summary_groups=sum([v for k, v in locals().items() if k.endswith('_summary_groups')], []),
)
