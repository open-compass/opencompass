from mmengine.config import read_base

with read_base():
    from .groups.agieval import agieval_summary_groups
    from .groups.mmlu import mmlu_summary_groups
    from .groups.cmmlu import cmmlu_summary_groups
    from .groups.ceval import ceval_summary_groups
    from .groups.bbh import bbh_summary_groups
    from .groups.GaokaoBench import GaokaoBench_summary_groups
    from .groups.flores import flores_summary_groups
    from .groups.jigsaw_multilingual import jigsaw_multilingual_summary_groups
    from .groups.tydiqa import tydiqa_summary_groups
    from .groups.xiezhi import xiezhi_summary_groups

summarizer = dict(
    dataset_abbrs=[
        '--------- 考试 Exam ---------',  # category
        # 'Mixed', # subcategory
        'ceval',
        'agieval',
        'mmlu',
        'GaokaoBench',
        'ARC-c',
        '--------- 语言 Language ---------',  # category
        # '字词释义', # subcategory
        'WiC',
        'summedits',
        # '成语习语', # subcategory
        'chid-dev',
        # '语义相似度', # subcategory
        'afqmc-dev',
        'bustm-dev',
        # '指代消解', # subcategory
        'cluewsc-dev',
        'WSC',
        'winogrande',
        # '翻译', # subcategory
        'flores_100',
        '--------- 知识 Knowledge ---------',  # category
        # '知识问答', # subcategory
        'BoolQ',
        'commonsense_qa',
        'nq',
        'triviaqa',
        # '多语种问答', # subcategory
        '--------- 推理 Reasoning ---------',  # category
        # '文本蕴含', # subcategory
        'cmnli',
        'ocnli',
        'ocnli_fc-dev',
        'AX_b',
        'AX_g',
        'CB',
        'RTE',
        # '常识推理', # subcategory
        'story_cloze',
        'COPA',
        'ReCoRD',
        'hellaswag',
        'piqa',
        'siqa',
        'strategyqa',
        # '数学推理', # subcategory
        'math',
        'gsm8k',
        # '定理应用', # subcategory
        'TheoremQA',
        # '代码', # subcategory
        'openai_humaneval',
        'mbpp',
        # '综合推理', # subcategory
        'bbh',
        '--------- 理解 Understanding ---------',  # category
        # '阅读理解', # subcategory
        'C3',
        'CMRC_dev',
        'DRCD_dev',
        'MultiRC',
        'race-middle',
        'race-high',
        'openbookqa_fact',
        # '内容总结', # subcategory
        'csl_dev',
        'lcsts',
        'Xsum',
        # '内容分析', # subcategory
        'eprstmt-dev',
        'lambada',
        'tnews-dev',
    ],
    summary_groups=sum(
        [v for k, v in locals().items() if k.endswith('_summary_groups')], []),
)
