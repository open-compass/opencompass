from mmengine.config import read_base

with read_base():
    from .groups.mmlu import mmlu_summary_groups
    from .groups.ceval import ceval_summary_groups
    from .groups.bbh import bbh_summary_groups
    from .groups.GaokaoBench import GaokaoBench_summary_groups
    from .groups.cmmlu import cmmlu_summary_groups

summarizer = dict(
    dataset_abbrs=[
        ['mmlu', 'naive_average'],
        ['cmmlu', 'naive_average'],
        ['ceval-test', 'naive_average'],
        ['GaokaoBench', 'weighted_average'],
        ['triviaqa_wiki_1shot', 'score'],
        ['nq_open_1shot', 'score'],
        ['race-high', 'accuracy'],
        ['winogrande', 'accuracy'],
        ['hellaswag', 'accuracy'],
        ['bbh', 'naive_average'],
        ['gsm8k', 'accuracy'],
        ['math', 'accuracy'],
        ['TheoremQA', 'accuracy'],
        ['openai_humaneval', 'humaneval_pass@1'],
        ['sanitized_mbpp', 'score'],
    ],
    summary_groups=sum(
        [v for k, v in locals().items() if k.endswith('_summary_groups')], []),
)
