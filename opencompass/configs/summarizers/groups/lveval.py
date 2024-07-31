len_levels = ['16k', '32k', '64k', '128k', '256k']

subsets_lveval_loogle_SD_mixup = [
    'LVEval_loogle_SD_mixup' + '_' + len_level for len_level in len_levels
]
subsets_lveval_cmrc_mixup = [
    'LVEval_cmrc_mixup' + '_' + len_level for len_level in len_levels
]
subsets_lveval_multifieldqa_en_mixup = [
    'LVEval_multifieldqa_en_mixup' + '_' + len_level
    for len_level in len_levels
]
subsets_lveval_multifieldqa_zh_mixup = [
    'LVEval_multifieldqa_zh_mixup' + '_' + len_level
    for len_level in len_levels
]
subsets_lveval_dureader_mixup = [
    'LVEval_dureader_mixup' + '_' + len_level for len_level in len_levels
]
subsets_lveval_loogle_CR_mixup = [
    'LVEval_loogle_CR_mixup' + '_' + len_level for len_level in len_levels
]
subsets_lveval_loogle_MIR_mixup = [
    'LVEval_loogle_MIR_mixup' + '_' + len_level for len_level in len_levels
]
subsets_lveval_hotpotwikiqa_mixup = [
    'LVEval_hotpotwikiqa_mixup' + '_' + len_level for len_level in len_levels
]
subsets_lveval_lic_mixup = [
    'LVEval_lic_mixup' + '_' + len_level for len_level in len_levels
]
subsets_lveval_factrecall_en = [
    'LVEval_factrecall_en' + '_' + len_level for len_level in len_levels
]
subsets_lveval_factrecall_zh = [
    'LVEval_factrecall_zh' + '_' + len_level for len_level in len_levels
]

subsets_lveval_single_hop_qa = (
    subsets_lveval_loogle_SD_mixup + subsets_lveval_cmrc_mixup
)
subsets_lveval_single_hop_cqa = (
    subsets_lveval_multifieldqa_en_mixup + subsets_lveval_multifieldqa_zh_mixup
)
subsets_lveval_multi_hop_qa = (
    subsets_lveval_dureader_mixup
    + subsets_lveval_loogle_CR_mixup
    + subsets_lveval_loogle_MIR_mixup
)
subsets_lveval_multi_hop_cqa = (
    subsets_lveval_hotpotwikiqa_mixup + subsets_lveval_lic_mixup
)
subsets_lveval_factrecall_cqa = (
    subsets_lveval_factrecall_en + subsets_lveval_factrecall_zh
)

subsets_lveval_qa = (
    subsets_lveval_single_hop_qa
    + subsets_lveval_single_hop_cqa
    + subsets_lveval_multi_hop_qa
    + subsets_lveval_multi_hop_cqa
    + subsets_lveval_factrecall_cqa
)

lveval_summary_groups = [
    {
        'name': 'LVEval_loogle_SD_mixup',
        'subsets': subsets_lveval_loogle_SD_mixup,
    },
    {'name': 'LVEval_cmrc_mixup', 'subsets': subsets_lveval_cmrc_mixup},
    {
        'name': 'LVEval_multifieldqa_en_mixup',
        'subsets': subsets_lveval_multifieldqa_en_mixup,
    },
    {
        'name': 'LVEval_multifieldqa_zh_mixup',
        'subsets': subsets_lveval_multifieldqa_zh_mixup,
    },
    {
        'name': 'LVEval_dureader_mixup',
        'subsets': subsets_lveval_dureader_mixup,
    },
    {
        'name': 'LVEval_loogle_CR_mixup',
        'subsets': subsets_lveval_loogle_CR_mixup,
    },
    {
        'name': 'LVEval_loogle_MIR_mixup',
        'subsets': subsets_lveval_loogle_MIR_mixup,
    },
    {
        'name': 'LVEval_hotpotwikiqa_mixup',
        'subsets': subsets_lveval_hotpotwikiqa_mixup,
    },
    {'name': 'LVEval_lic_mixup', 'subsets': subsets_lveval_lic_mixup},
    {'name': 'LVEval_factrecall_en', 'subsets': subsets_lveval_factrecall_en},
    {'name': 'LVEval_factrecall_zh', 'subsets': subsets_lveval_factrecall_zh},
    {'name': 'LVEval_single_hop_qa', 'subsets': subsets_lveval_single_hop_qa},
    {
        'name': 'LVEval_single_hop_cqa',
        'subsets': subsets_lveval_single_hop_cqa,
    },
    {'name': 'LVEval_multi_hop_qa', 'subsets': subsets_lveval_multi_hop_qa},
    {'name': 'LVEval_multi_hop_cqa', 'subsets': subsets_lveval_multi_hop_cqa},
    {
        'name': 'LVEval_factrecall_cqa',
        'subsets': subsets_lveval_factrecall_cqa,
    },
    {'name': 'LVEval_qa', 'subsets': subsets_lveval_qa},
]
