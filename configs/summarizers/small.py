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

summarizer = dict(
    dataset_abbrs = [
        '--- Exam ---',
        'mmlu',
        'ceval',
        'bbh',
        '--- ChineseUniversal ---',
        'CMRC_dev',
        'DRCD_dev',
        'afqmc-dev',
        'bustm-dev',
        'chid-dev',
        'cluewsc-dev',
        'eprstmt-dev',
        '--- Coding ---',
        'openai_humaneval',
        'mbpp',
        '--- Completion ---',
        'lambada',
        'story_cloze',
        '--- EnglishUniversal ---',
        'AX_b',
        'AX_g',
        'BoolQ',
        'CB',
        'COPA',
        'MultiRC',
        'RTE',
        'ReCoRD',
        'WiC',
        'WSC',
        'race-high',
        'race-middle',
        '--- Reasoning ---',
        'math',
        'gsm8k',
        'summedits',
        '--- QA ---',
        'hellaswag',
        'piqa',
        'winogrande',
        'openbookqa',
        'openbookqa_fact',
        'nq',
        'triviaqa',
    ],
    summary_groups=sum([v for k, v in locals().items() if k.endswith('_summary_groups')], []),
)
