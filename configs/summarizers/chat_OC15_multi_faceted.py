from mmengine.config import read_base
from opencompass.summarizers import MultiFacetedSummarizer

with read_base():
    from .groups.mmlu import mmlu_summary_groups
    from .groups.cmmlu import cmmlu_summary_groups
    from .groups.ceval import ceval_summary_groups
    from .groups.bbh import bbh_summary_groups
    from .groups.GaokaoBench import GaokaoBench_summary_groups

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

overall_dataset_abbrs = [
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
]

mmlu_summary_groups_dict = {g['name']: g['subsets'] for g in mmlu_summary_groups}
mmlu_dataset_abbrs = [
    ['mmlu', 'naive_average'],
    ['mmlu-stem', 'naive_average'],
    ['mmlu-social-science', 'naive_average'],
    ['mmlu-humanities', 'naive_average'],
    ['mmlu-other', 'naive_average'],
    *mmlu_summary_groups_dict['mmlu-stem'],
    *mmlu_summary_groups_dict['mmlu-social-science'],
    *mmlu_summary_groups_dict['mmlu-humanities'],
    *mmlu_summary_groups_dict['mmlu-other'],
]

cmmlu_summary_groups_dict = {g['name']: g['subsets'] for g in cmmlu_summary_groups}
cmmlu_dataset_abbrs = [
    ['cmmlu', 'naive_average'],
    ['cmmlu-stem', 'naive_average'],
    ['cmmlu-social-science', 'naive_average'],
    ['cmmlu-humanities', 'naive_average'],
    ['cmmlu-other', 'naive_average'],
    ['cmmlu-china-specific', 'naive_average'],
    *cmmlu_summary_groups_dict['cmmlu-stem'],
    *cmmlu_summary_groups_dict['cmmlu-social-science'],
    *cmmlu_summary_groups_dict['cmmlu-humanities'],
    *cmmlu_summary_groups_dict['cmmlu-other'],
]

ceval_summary_groups_dict = {g['name']: g['subsets'] for g in ceval_summary_groups}
ceval_dataset_abbrs = [
    ['ceval', 'naive_average'],
    ['ceval-stem', 'naive_average'],
    ['ceval-social-science', 'naive_average'],
    ['ceval-humanities', 'naive_average'],
    ['ceval-other', 'naive_average'],
    ['ceval-hard', 'naive_average'],
    *ceval_summary_groups_dict['ceval-stem'],
    *ceval_summary_groups_dict['ceval-social-science'],
    *ceval_summary_groups_dict['ceval-humanities'],
    *ceval_summary_groups_dict['ceval-other'],
]

bbh_summary_groups_dict = {g['name']: g['subsets'] for g in bbh_summary_groups}
bbh_dataset_abbrs = [
    ['bbh', 'naive_average'],
    *bbh_summary_groups_dict['bbh'],
]

GaokaoBench_summary_groups_dict = {g['name']: g['subsets'] for g in GaokaoBench_summary_groups}
GaokaoBench_dataset_abbrs = [
    ['GaokaoBench', 'weighted_average'],
    *GaokaoBench_summary_groups_dict['GaokaoBench'],
]

sanitized_mbpp_dataset_abbrs = [
    ['sanitized_mbpp', 'score'],
    ['sanitized_mbpp', 'pass'],
    ['sanitized_mbpp', 'failed'],
    ['sanitized_mbpp', 'wrong_answer'],
    ['sanitized_mbpp', 'timeout'],
]

IFEval_dataset_abbrs = [
    ['IFEval', 'Prompt-level-strict-accuracy'],
    ['IFEval', 'Inst-level-strict-accuracy'],
    ['IFEval', 'Prompt-level-loose-accuracy'],
    ['IFEval', 'Inst-level-loose-accuracy'],
]

summarizer = dict(
    type=MultiFacetedSummarizer,
    dataset_abbrs_list=[
        {'name': 'overall', 'dataset_abbrs': overall_dataset_abbrs},
        {'name': 'mmlu', 'dataset_abbrs': mmlu_dataset_abbrs},
        {'name': 'cmmlu', 'dataset_abbrs': cmmlu_dataset_abbrs},
        {'name': 'ceval', 'dataset_abbrs': ceval_dataset_abbrs},
        {'name': 'bbh', 'dataset_abbrs': bbh_dataset_abbrs},
        {'name': 'GaokaoBench', 'dataset_abbrs': GaokaoBench_dataset_abbrs},
        {'name': 'sanitized_mbpp', 'dataset_abbrs': sanitized_mbpp_dataset_abbrs},
        {'name': 'triviaqa', 'dataset_abbrs': [['triviaqa_wiki_1shot', 'score']]},
        {'name': 'nq', 'dataset_abbrs': [['nq_open_1shot', 'score']]},
        {'name': 'race', 'dataset_abbrs': [['race-high', 'accuracy']]},
        {'name': 'winogrande', 'dataset_abbrs': [['winogrande', 'accuracy']]},
        {'name': 'hellaswag', 'dataset_abbrs': [['hellaswag', 'accuracy']]},
        {'name': 'gsm8k', 'dataset_abbrs': [['gsm8k', 'accuracy']]},
        {'name': 'math', 'dataset_abbrs': [['math', 'accuracy']]},
        {'name': 'TheoremQA', 'dataset_abbrs': [['TheoremQA', 'score']]},
        {'name': 'humaneval', 'dataset_abbrs': [['openai_humaneval', 'humaneval_pass@1']]},
        {'name': 'GPQA', 'dataset_abbrs': [['GPQA_diamond', 'accuracy']]},
        {'name': 'IFEval', 'dataset_abbrs': IFEval_dataset_abbrs},
    ],
    summary_groups=sum([v for k, v in locals().items() if k.endswith('_summary_groups')], []),
)
