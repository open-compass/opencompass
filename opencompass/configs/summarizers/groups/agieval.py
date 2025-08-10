agieval_summary_groups = []

_agieval_chinese_sets = ['gaokao-chinese', 'gaokao-english', 'gaokao-geography', 'gaokao-history', 'gaokao-biology', 'gaokao-chemistry', 'gaokao-physics', 'gaokao-mathqa', 'logiqa-zh', 'jec-qa-kd', 'jec-qa-ca', 'gaokao-mathcloze']
_agieval_chinese_sets = ['agieval-' + s for s in _agieval_chinese_sets]
agieval_summary_groups.append({'name': 'agieval-chinese', 'subsets': _agieval_chinese_sets})

_agieval_english_sets = ['lsat-ar', 'lsat-lr', 'lsat-rc', 'logiqa-en', 'sat-math', 'sat-en', 'sat-en-without-passage', 'aqua-rat', 'math']
_agieval_english_sets = ['agieval-' + s for s in _agieval_english_sets]
agieval_summary_groups.append({'name': 'agieval-english', 'subsets': _agieval_english_sets})

_agieval_gaokao_sets = ['gaokao-chinese', 'gaokao-english', 'gaokao-geography', 'gaokao-history', 'gaokao-biology', 'gaokao-chemistry', 'gaokao-physics', 'gaokao-mathqa', 'gaokao-mathcloze']
_agieval_gaokao_sets = ['agieval-' + s for s in _agieval_gaokao_sets]
agieval_summary_groups.append({'name': 'agieval-gaokao', 'subsets': _agieval_gaokao_sets})

_agieval_all = ['gaokao-chinese', 'gaokao-english', 'gaokao-geography', 'gaokao-history', 'gaokao-biology', 'gaokao-chemistry', 'gaokao-physics', 'gaokao-mathqa', 'logiqa-zh', 'lsat-ar', 'lsat-lr', 'lsat-rc', 'logiqa-en', 'sat-math', 'sat-en', 'sat-en-without-passage', 'aqua-rat', 'jec-qa-kd', 'jec-qa-ca', 'gaokao-mathcloze', 'math']
_agieval_all = ['agieval-' + s for s in _agieval_all]
agieval_summary_groups.append({'name': 'agieval', 'subsets': _agieval_all})
