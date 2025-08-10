tydiqa_summary_groups = []

_tydiqa = ['arabic', 'bengali', 'english', 'finnish', 'indonesian', 'japanese', 'korean', 'russian', 'swahili', 'telugu', 'thai']
_tydiqa = ['tydiqa-goldp_' + s for s in _tydiqa]
tydiqa_summary_groups.append({'name': 'tydiqa-goldp', 'subsets': _tydiqa})
