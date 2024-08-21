ceval_summary_groups = []

_ceval_stem = ['computer_network', 'operating_system', 'computer_architecture', 'college_programming', 'college_physics', 'college_chemistry', 'advanced_mathematics', 'probability_and_statistics', 'discrete_mathematics', 'electrical_engineer', 'metrology_engineer', 'high_school_mathematics', 'high_school_physics', 'high_school_chemistry', 'high_school_biology', 'middle_school_mathematics', 'middle_school_biology', 'middle_school_physics', 'middle_school_chemistry', 'veterinary_medicine']
_ceval_stem = ['ceval-' + s for s in _ceval_stem]
ceval_summary_groups.append({'name': 'ceval-stem', 'subsets': _ceval_stem})

_ceval_social_science = ['college_economics', 'business_administration', 'marxism', 'mao_zedong_thought', 'education_science', 'teacher_qualification', 'high_school_politics', 'high_school_geography', 'middle_school_politics', 'middle_school_geography']
_ceval_social_science = ['ceval-' + s for s in _ceval_social_science]
ceval_summary_groups.append({'name': 'ceval-social-science', 'subsets': _ceval_social_science})

_ceval_humanities = ['modern_chinese_history', 'ideological_and_moral_cultivation', 'logic', 'law', 'chinese_language_and_literature', 'art_studies', 'professional_tour_guide', 'legal_professional', 'high_school_chinese', 'high_school_history', 'middle_school_history']
_ceval_humanities = ['ceval-' + s for s in _ceval_humanities]
ceval_summary_groups.append({'name': 'ceval-humanities', 'subsets': _ceval_humanities})

_ceval_other = ['civil_servant', 'sports_science', 'plant_protection', 'basic_medicine', 'clinical_medicine', 'urban_and_rural_planner', 'accountant', 'fire_engineer', 'environmental_impact_assessment_engineer', 'tax_accountant', 'physician']
_ceval_other = ['ceval-' + s for s in _ceval_other]
ceval_summary_groups.append({'name': 'ceval-other', 'subsets': _ceval_other})

_ceval_hard = ['advanced_mathematics', 'discrete_mathematics', 'probability_and_statistics', 'college_chemistry', 'college_physics', 'high_school_mathematics', 'high_school_chemistry', 'high_school_physics']
_ceval_hard = ['ceval-' + s for s in _ceval_hard]
ceval_summary_groups.append({'name': 'ceval-hard', 'subsets': _ceval_hard})

_ceval_all = _ceval_stem + _ceval_social_science + _ceval_humanities + _ceval_other
ceval_summary_groups.append({'name': 'ceval', 'subsets': _ceval_all})

_ceval_stem = ['computer_network', 'operating_system', 'computer_architecture', 'college_programming', 'college_physics', 'college_chemistry', 'advanced_mathematics', 'probability_and_statistics', 'discrete_mathematics', 'electrical_engineer', 'metrology_engineer', 'high_school_mathematics', 'high_school_physics', 'high_school_chemistry', 'high_school_biology', 'middle_school_mathematics', 'middle_school_biology', 'middle_school_physics', 'middle_school_chemistry', 'veterinary_medicine']
_ceval_stem = ['ceval-test-' + s for s in _ceval_stem]
ceval_summary_groups.append({'name': 'ceval-test-stem', 'subsets': _ceval_stem})

_ceval_social_science = ['college_economics', 'business_administration', 'marxism', 'mao_zedong_thought', 'education_science', 'teacher_qualification', 'high_school_politics', 'high_school_geography', 'middle_school_politics', 'middle_school_geography']
_ceval_social_science = ['ceval-test-' + s for s in _ceval_social_science]
ceval_summary_groups.append({'name': 'ceval-test-social-science', 'subsets': _ceval_social_science})

_ceval_humanities = ['modern_chinese_history', 'ideological_and_moral_cultivation', 'logic', 'law', 'chinese_language_and_literature', 'art_studies', 'professional_tour_guide', 'legal_professional', 'high_school_chinese', 'high_school_history', 'middle_school_history']
_ceval_humanities = ['ceval-test-' + s for s in _ceval_humanities]
ceval_summary_groups.append({'name': 'ceval-test-humanities', 'subsets': _ceval_humanities})

_ceval_other = ['civil_servant', 'sports_science', 'plant_protection', 'basic_medicine', 'clinical_medicine', 'urban_and_rural_planner', 'accountant', 'fire_engineer', 'environmental_impact_assessment_engineer', 'tax_accountant', 'physician']
_ceval_other = ['ceval-test-' + s for s in _ceval_other]
ceval_summary_groups.append({'name': 'ceval-test-other', 'subsets': _ceval_other})

_ceval_hard = ['advanced_mathematics', 'discrete_mathematics', 'probability_and_statistics', 'college_chemistry', 'college_physics', 'high_school_mathematics', 'high_school_chemistry', 'high_school_physics']
_ceval_hard = ['ceval-test-' + s for s in _ceval_hard]
ceval_summary_groups.append({'name': 'ceval-test-hard', 'subsets': _ceval_hard})

_ceval_all = _ceval_stem + _ceval_social_science + _ceval_humanities + _ceval_other
ceval_summary_groups.append({'name': 'ceval-test', 'subsets': _ceval_all})
