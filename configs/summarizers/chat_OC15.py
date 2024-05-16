from mmengine.config import read_base

with read_base():
    from .groups.mmlu import mmlu_summary_groups
    from .groups.cmmlu import cmmlu_summary_groups
    from .groups.ceval import ceval_summary_groups
    from .groups.bbh import bbh_summary_groups
    from .groups.GaokaoBench import GaokaoBench_summary_groups
    from .groups.lcbench import lcbench_summary_groups

other_summary_groups = [
    {
        'name': 'average',
        'subsets': [
            ['mmlu', 'naive_average'],
            ['cmmlu', 'naive_average'],
            ['ceval', 'naive_average'],
            ['GaokaoBench', 'weighted_average'],
            ['triviaqa_wiki_1shot', 'score'],
            ['nq_open_1shot', 'score'],
            ['race-high', 'accuracy'],
            ['winogrande', 'accuracy'],
            ['hellaswag', 'accuracy'],
            ['bbh', 'naive_average'],
            ['gsm8k', 'accuracy'],
            ['math', 'accuracy'],
            ['TheoremQA', 'score'],
            ['openai_humaneval', 'humaneval_pass@1'],
            ['sanitized_mbpp', 'score'],
            ['GPQA_diamond', 'accuracy'],
            ['IFEval', 'Prompt-level-strict-accuracy'],
        ],
    },
]

summarizer = dict(
    dataset_abbrs=[
        ['average', 'naive_average'],
        ['mmlu', 'naive_average'],
        ['cmmlu', 'naive_average'],
        ['ceval', 'naive_average'],
        ['GaokaoBench', 'weighted_average'],
        ['triviaqa_wiki_1shot', 'score'],
        ['nq_open_1shot', 'score'],
        ['race-high', 'accuracy'],
        ['winogrande', 'accuracy'],
        ['hellaswag', 'accuracy'],
        ['bbh', 'naive_average'],
        ['gsm8k', 'accuracy'],
        ['math', 'accuracy'],
        ['TheoremQA', 'score'],
        ['openai_humaneval', 'humaneval_pass@1'],
        ['sanitized_mbpp', 'score'],
        ['GPQA_diamond', 'accuracy'],
        ['IFEval', 'Prompt-level-strict-accuracy'],

        '',

        'mmlu',
        'mmlu-stem',
        'mmlu-social-science',
        'mmlu-humanities',
        'mmlu-other',

        'cmmlu',
        'cmmlu-stem',
        'cmmlu-social-science',
        'cmmlu-humanities',
        'cmmlu-other',
        'cmmlu-china-specific',

        'ceval',
        'ceval-stem',
        'ceval-social-science',
        'ceval-humanities',
        'ceval-other',
        'ceval-hard',
    ],
    summary_groups=sum(
        [v for k, v in locals().items() if k.endswith('_summary_groups')], []),
)
