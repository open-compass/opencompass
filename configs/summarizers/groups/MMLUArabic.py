sub_categories = {
    'math': ['abstract_algebra', 'college_mathematics', 'elementary_mathematics', 'high_school_mathematics', 'high_school_statistics'],
    'health': ['anatomy', 'clinical_knowledge', 'college_medicine', 'human_aging', 'medical_genetics', 'nutrition', 'professional_medicine', 'virology'],
    'physics': ['astronomy', 'college_physics', 'conceptual_physics', 'high_school_physics'],
    'business': ['business_ethics', 'management', 'marketing'],
    'biology': ['college_biology', 'high_school_biology'],
    'chemistry': ['college_chemistry', 'high_school_chemistry'],
    'computer science': ['college_computer_science', 'computer_security', 'high_school_computer_science', 'machine_learning'],
    'economics': ['econometrics', 'high_school_macroeconomics', 'high_school_microeconomics'],
    'engineering': ['electrical_engineering'],
    'philosophy': ['formal_logic', 'logical_fallacies', 'moral_disputes', 'moral_scenarios', 'philosophy', 'world_religions'],
    'other': ['global_facts', 'miscellaneous', 'professional_accounting'],
    'history': ['high_school_european_history', 'high_school_us_history', 'high_school_world_history', 'prehistory'],
    'geography': ['high_school_geography'],
    'politics': ['high_school_government_and_politics', 'public_relations', 'security_studies', 'us_foreign_policy'],
    'psychology': ['high_school_psychology', 'professional_psychology'],
    'culture': ['human_sexuality', 'sociology'],
    'law': ['international_law', 'jurisprudence', 'professional_law']
}

categories = {
    'STEM': ['physics', 'chemistry', 'biology', 'computer science', 'math', 'engineering'],
    'humanities': ['history', 'philosophy', 'law'],
    'social_sciences': ['politics', 'culture', 'economics', 'geography', 'psychology'],
    'other': ['other', 'business', 'health'],
}

category2subject = {}
for k, v in categories.items():
    for subject, subcat in sub_categories.items():
        if subject in v:
            for c in subcat:
                category2subject.setdefault(k, []).append(c)

MMLUArabic_summary_groups = []

_MMLUArabic_stem = ['acegpt_MMLUArabic_' + s for s in category2subject['STEM']]
MMLUArabic_summary_groups.append({'name': 'acegpt_MMLUArabic_STEM', 'subsets': _MMLUArabic_stem})

_MMLUArabic_humanities = ['acegpt_MMLUArabic_' + s for s in category2subject['humanities']]
MMLUArabic_summary_groups.append({'name': 'acegpt_MMLUArabic_humanities', 'subsets': _MMLUArabic_humanities})

_MMLUArabic_social_science = ['acegpt_MMLUArabic_' + s for s in category2subject['social_sciences']]
MMLUArabic_summary_groups.append({'name': 'acegpt_MMLUArabic_social_science', 'subsets': _MMLUArabic_social_science})

_MMLUArabic_other = ['acegpt_MMLUArabic_' + s for s in category2subject['other']]
MMLUArabic_summary_groups.append({'name': 'acegpt_MMLUArabic_other', 'subsets': _MMLUArabic_other})

_MMLUArabic_all = _MMLUArabic_stem + _MMLUArabic_humanities + _MMLUArabic_social_science + _MMLUArabic_other
MMLUArabic_summary_groups.append({'name': 'acegpt_MMLUArabic', 'subsets': _MMLUArabic_all})
