GaokaoBench_summary_groups = []

# gaokao-bench
_GaokaoBench_weights = {'2010-2022_Math_II_MCQs': 1090, '2010-2022_Math_I_MCQs': 1070, '2010-2022_History_MCQs': 1148, '2010-2022_Biology_MCQs': 900, '2010-2022_Political_Science_MCQs': 1280, '2010-2022_Physics_MCQs': 384, '2010-2022_Chemistry_MCQs': 744, '2010-2013_English_MCQs': 105, '2010-2022_Chinese_Modern_Lit': 261, '2010-2022_English_Fill_in_Blanks': 900.0, '2012-2022_English_Cloze_Test': 260, '2010-2022_Geography_MCQs': 380, '2010-2022_English_Reading_Comp': 940, '2010-2022_Chinese_Lang_and_Usage_MCQs': 240}
_GaokaoBench_weights = {'GaokaoBench_' + k: v for k, v in _GaokaoBench_weights.items()}
GaokaoBench_summary_groups.append({'name': 'GaokaoBench', 'subsets': list(_GaokaoBench_weights.keys()), 'weights': _GaokaoBench_weights})
