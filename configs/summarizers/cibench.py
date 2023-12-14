from mmengine.config import read_base

with read_base():
    from .groups.cibench import cibench_summary_groups

summarizer = dict(
    dataset_abbrs=[
        '######## CIBench Generation ########', # category
        ['cibench', 'executable'],
        ['cibench', 'general_correct'],
        ['cibench', 'vis_sim'],
        '######## CIBench Template ########', # category
        'cibench_template:executable',
        'cibench_template:numeric_correct',
        'cibench_template:text_score',
        'cibench_template:vis_sim',
        '######## CIBench Template Chinese ########', # category
        'cibench_template_cn:executable',
        'cibench_template_cn:numeric_correct',
        'cibench_template_cn:text_score',
        'cibench_template_cn:vis_sim',
    ],
    summary_groups=sum(
        [v for k, v in locals().items() if k.endswith("_summary_groups")], [])
)
