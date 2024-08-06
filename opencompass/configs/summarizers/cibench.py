from mmengine.config import read_base

with read_base():
    from .groups.cibench import cibench_summary_groups

summarizer = dict(
    dataset_abbrs=[
        '######## CIBench Generation########', # category
        'cibench_generation:tool_rate',
        'cibench_generation:executable',
        'cibench_generation:numeric_correct',
        'cibench_generation:text_score',
        'cibench_generation:vis_sim',
        '######## CIBench Generation Oracle########', # category
        'cibench_generation_oracle:tool_rate',
        'cibench_generation_oracle:executable',
        'cibench_generation_oracle:numeric_correct',
        'cibench_generation_oracle:text_score',
        'cibench_generation_oracle:vis_sim',
        '######## CIBench Template ########', # category
        'cibench_template:tool_rate',
        'cibench_template:executable',
        'cibench_template:numeric_correct',
        'cibench_template:text_score',
        'cibench_template:vis_sim',
        '######## CIBench Template Oracle########', # category
        'cibench_template_oracle:tool_rate',
        'cibench_template_oracle:executable',
        'cibench_template_oracle:numeric_correct',
        'cibench_template_oracle:text_score',
        'cibench_template_oracle:vis_sim',
        '######## CIBench Template Chinese ########', # category
        'cibench_template_cn:tool_rate',
        'cibench_template_cn:executable',
        'cibench_template_cn:numeric_correct',
        'cibench_template_cn:text_score',
        'cibench_template_cn:vis_sim',
        '######## CIBench Template Chinese Oracle########', # category
        'cibench_template_cn_oracle:tool_rate',
        'cibench_template_cn_oracle:executable',
        'cibench_template_cn_oracle:numeric_correct',
        'cibench_template_cn_oracle:text_score',
        'cibench_template_cn_oracle:vis_sim',
        '######## CIBench Category Metric ########',
        'cibench_data_manipulation:scores',
        'cibench_data_visualization:scores',
        'cibench_modeling:scores',
        'cibench_nlp:scores',
        'cibench_ip:scores',
        'cibench_math:scores',
        '######## CIBench Category Metric Oracle ########',
        'cibench_data_manipulation_oracle:scores',
        'cibench_data_visualization_oracle:scores',
        'cibench_modeling_oracle:scores',
        'cibench_nlp_oracle:scores',
        'cibench_ip_oracle:scores',
        'cibench_math_oracle:scores',

    ],
    summary_groups=sum(
        [v for k, v in locals().items() if k.endswith('_summary_groups')], [])
)
