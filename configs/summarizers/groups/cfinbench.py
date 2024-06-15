mapping_1 = {"1-1": "政治经济学",
             "1-2": "西方经济学",
             "1-3": "微观经济学",
             "1-4": "宏观经济学",
             "1-5": "产业经济学",
             "1-6": "财政学",
             "1-7": "国际贸易学",
             "1-8": "统计学",
             "1-9": "审计学",
             "1-10": "经济史",
             "1-11": "金融学"}

mapping_2 = {"2-1": "税务从业资格",
             "2-2": "期货从业资格",
             "2-3": "基金从业资格",
             "2-4": "地产从业资格",
             "2-5": "保险从业资格",
             "2-6": "证券从业资格",
             "2-7": "银行从业资格",
             "2-8": "注册会计师"}

mapping_3 = {"3-1": "初级审计师",
             "3-2": "中级审计师",
             "3-3": "初级统计师",
             "3-4": "中级统计师",
             "3-5": "初级经济师",
             "3-6": "中级经济师",
             "3-7": "初级银行从业人员",
             "3-8": "中级银行从业人员",
             "3-9": "初级会计师",
             "3-10": "中级会计师",
             "3-11": "税务师",
             "3-12": "资产评估师",
             "3-13": "证券分析师"}

mapping_4 = {"4-1": "税法一",
             "4-2": "税法二",
             "4-3": "税务稽查",
             "4-4": "商业法",
             "4-5": "证券法",
             "4-6": "保险法",
             "4-7": "经济法",
             "4-8": "银行业法",
             "4-9": "期货法",
             "4-10": "金融法",
             "4-11": "民法"}

cfinbench_summary_groups = []


_cfinbench_single_1_zero_shot = ["CFinBenchval" + _s + '-' + "gen-单选题" + '_zero-shot' for _s in mapping_1]
_cfinbench_single_2_zero_shot = ["CFinBenchval" + _s + '-' + "gen-单选题" + '_zero-shot' for _s in mapping_2]
_cfinbench_single_3_zero_shot = ["CFinBenchval" + _s + '-' + "gen-单选题" + '_zero-shot' for _s in mapping_3]
_cfinbench_single_4_zero_shot = ["CFinBenchval" + _s + '-' + "gen-单选题" + '_zero-shot' for _s in mapping_4]

_cfinbench_multi_1_zero_shot = ["CFinBenchval" + _s + '-' + "多选题" + '_zero-shot' for _s in mapping_1]
_cfinbench_multi_2_zero_shot = ["CFinBenchval" + _s + '-' + "多选题" + '_zero-shot' for _s in mapping_2]
_cfinbench_multi_3_zero_shot = ["CFinBenchval" + _s + '-' + "多选题" + '_zero-shot' for _s in mapping_3]
_cfinbench_multi_4_zero_shot = ["CFinBenchval" + _s + '-' + "多选题" + '_zero-shot' for _s in mapping_4]

_cfinbench_judgment_1_zero_shot = ["CFinBenchval" + _s + '-' + "判断题" + '_zero-shot' for _s in mapping_1]
_cfinbench_judgment_2_zero_shot = ["CFinBenchval" + _s + '-' + "判断题" + '_zero-shot' for _s in mapping_2]
_cfinbench_judgment_3_zero_shot = ["CFinBenchval" + _s + '-' + "判断题" + '_zero-shot' for _s in mapping_3]
_cfinbench_judgment_4_zero_shot = ["CFinBenchval" + _s + '-' + "判断题" + '_zero-shot' for _s in mapping_4]

_cfinbench_single_1_few_shot = ["CFinBenchval" + _s + '-' + "gen-单选题" + '_few-shot' for _s in mapping_1]
_cfinbench_single_2_few_shot = ["CFinBenchval" + _s + '-' + "gen-单选题" + '_few-shot' for _s in mapping_2]
_cfinbench_single_3_few_shot = ["CFinBenchval" + _s + '-' + "gen-单选题" + '_few-shot' for _s in mapping_3]
_cfinbench_single_4_few_shot = ["CFinBenchval" + _s + '-' + "gen-单选题" + '_few-shot' for _s in mapping_4]

_cfinbench_multi_1_few_shot = ["CFinBenchval" + _s + '-' + "多选题" + '_few-shot' for _s in mapping_1]
_cfinbench_multi_2_few_shot = ["CFinBenchval" + _s + '-' + "多选题" + '_few-shot' for _s in mapping_2]
_cfinbench_multi_3_few_shot = ["CFinBenchval" + _s + '-' + "多选题" + '_few-shot' for _s in mapping_3]
_cfinbench_multi_4_few_shot = ["CFinBenchval" + _s + '-' + "多选题" + '_few-shot' for _s in mapping_4]

_cfinbench_judgment_1_few_shot = ["CFinBenchval" + _s + '-' + "判断题" + '_few-shot' for _s in mapping_1]
_cfinbench_judgment_2_few_shot = ["CFinBenchval" + _s + '-' + "判断题" + '_few-shot' for _s in mapping_2]
_cfinbench_judgment_3_few_shot = ["CFinBenchval" + _s + '-' + "判断题" + '_few-shot' for _s in mapping_3]
_cfinbench_judgment_4_few_shot = ["CFinBenchval" + _s + '-' + "判断题" + '_few-shot' for _s in mapping_4]


_subject_zero_shot = _cfinbench_single_1_zero_shot + _cfinbench_multi_1_zero_shot + _cfinbench_judgment_1_zero_shot
_qualification_zero_shot = _cfinbench_single_2_zero_shot + _cfinbench_multi_2_zero_shot + _cfinbench_judgment_2_zero_shot
_practice_zero_shot = _cfinbench_single_3_zero_shot + _cfinbench_multi_3_zero_shot + _cfinbench_judgment_3_zero_shot
_law_zero_shot = _cfinbench_single_4_zero_shot + _cfinbench_multi_4_zero_shot + _cfinbench_judgment_4_zero_shot

_subject_few_shot = _cfinbench_single_1_few_shot + _cfinbench_multi_1_few_shot + _cfinbench_judgment_1_few_shot
_qualification_few_shot = _cfinbench_single_2_few_shot + _cfinbench_multi_2_few_shot + _cfinbench_judgment_2_few_shot
_practice_few_shot = _cfinbench_single_3_few_shot + _cfinbench_multi_3_few_shot + _cfinbench_judgment_3_few_shot
_law_few_shot = _cfinbench_single_4_few_shot + _cfinbench_multi_4_few_shot + _cfinbench_judgment_4_few_shot


_subject_zero_shot_weights = {k: (0.2 if "判断" in k else 0.4) for k in _subject_zero_shot}
_qualification_zero_shot_weights = {k: (0.2 if "判断" in k else 0.4) for k in _qualification_zero_shot}
_practice_zero_shot_weights = {k: (0.2 if "判断" in k else 0.4) for k in _practice_zero_shot}
_law_zero_shot_weights = {k: (0.2 if "判断" in k else 0.4) for k in _law_zero_shot}

_subject_few_shot_weights = {k: (0.2 if "判断" in k else 0.4) for k in _subject_few_shot}
_qualification_few_shot_weights = {k: (0.2 if "判断" in k else 0.4) for k in _qualification_few_shot}
_practice_few_shot_weights = {k: (0.2 if "判断" in k else 0.4) for k in _practice_few_shot}
_law_few_shot_weights = {k: (0.2 if "判断" in k else 0.4) for k in _law_few_shot}


cfinbench_summary_groups.append({'name': 'subject_weighted_zero_shot', 'subsets': _subject_zero_shot, "weights": _subject_zero_shot_weights})
cfinbench_summary_groups.append({'name': 'qualification_weighted_zero_shot', 'subsets': _qualification_zero_shot, "weights": _qualification_zero_shot_weights})
cfinbench_summary_groups.append({'name': 'practice_weighted_zero_shot', 'subsets': _practice_zero_shot, "weights": _practice_zero_shot_weights})
cfinbench_summary_groups.append({'name': 'law_weighted_zero_shot', 'subsets': _law_zero_shot, "weights": _law_zero_shot_weights})

cfinbench_summary_groups.append({'name': 'subject_weighted_few_shot', 'subsets': _subject_few_shot, "weights": _subject_few_shot_weights})
cfinbench_summary_groups.append({'name': 'qualification_weighted_few_shot', 'subsets': _qualification_few_shot, "weights": _qualification_few_shot_weights})
cfinbench_summary_groups.append({'name': 'practice_weighted_few_shot', 'subsets': _practice_few_shot, "weights": _practice_few_shot_weights})
cfinbench_summary_groups.append({'name': 'law_weighted_few_shot', 'subsets': _law_few_shot, "weights": _law_few_shot_weights})
