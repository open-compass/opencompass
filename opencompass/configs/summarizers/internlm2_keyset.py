from mmengine.config import read_base

with read_base():
    from .groups.agieval import agieval_summary_groups
    from .groups.mmlu import mmlu_summary_groups
    from .groups.bbh import bbh_summary_groups

summarizer = dict(
    dataset_abbrs=[
        ['mmlu', 'naive_average'],
        ['agieval', 'naive_average'],
        ['bbh', 'naive_average'],
        ['gsm8k', 'accuracy'],
        ['math', 'accuracy'],
        ['openai_humaneval', 'humaneval_pass@1'],
        ['sanitized_mbpp', 'score'],
    ],
    summary_groups=sum(
        [v for k, v in locals().items() if k.endswith('_summary_groups')], []),
)
