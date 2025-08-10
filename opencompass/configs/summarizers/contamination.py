from mmengine.config import read_base
from opencompass.summarizers import CircularSummarizer

with read_base():
    from .groups.ceval import ceval_summary_groups

ceval_category_weights = {
    'computer_network': {'accuracy - clean': 11, 'accuracy - input contaminated': 2, 'accuracy - input-and-label contaminated': 6, 'accuracy - not labeled': 0},
    'operating_system': {'accuracy - clean': 14, 'accuracy - input contaminated': 0, 'accuracy - input-and-label contaminated': 5, 'accuracy - not labeled': 0},
    'computer_architecture': {'accuracy - clean': 7, 'accuracy - input contaminated': 2, 'accuracy - input-and-label contaminated': 12, 'accuracy - not labeled': 0},
    'college_programming': {'accuracy - clean': 22, 'accuracy - input contaminated': 1, 'accuracy - input-and-label contaminated': 14, 'accuracy - not labeled': 0},
    'college_physics': {'accuracy - clean': 6, 'accuracy - input contaminated': 4, 'accuracy - input-and-label contaminated': 9, 'accuracy - not labeled': 0},
    'college_chemistry': {'accuracy - clean': 21, 'accuracy - input contaminated': 1, 'accuracy - input-and-label contaminated': 2, 'accuracy - not labeled': 0},
    'advanced_mathematics': {'accuracy - clean': 19, 'accuracy - input contaminated': 0, 'accuracy - input-and-label contaminated': 0, 'accuracy - not labeled': 0},
    'probability_and_statistics': {'accuracy - clean': 18, 'accuracy - input contaminated': 0, 'accuracy - input-and-label contaminated': 0, 'accuracy - not labeled': 0},
    'discrete_mathematics': {'accuracy - clean': 14, 'accuracy - input contaminated': 1, 'accuracy - input-and-label contaminated': 1, 'accuracy - not labeled': 0},
    'electrical_engineer': {'accuracy - clean': 18, 'accuracy - input contaminated': 4, 'accuracy - input-and-label contaminated': 15, 'accuracy - not labeled': 0},
    'metrology_engineer': {'accuracy - clean': 8, 'accuracy - input contaminated': 2, 'accuracy - input-and-label contaminated': 14, 'accuracy - not labeled': 0},
    'high_school_mathematics': {'accuracy - clean': 18, 'accuracy - input contaminated': 0, 'accuracy - input-and-label contaminated': 0, 'accuracy - not labeled': 0},
    'high_school_physics': {'accuracy - clean': 12, 'accuracy - input contaminated': 2, 'accuracy - input-and-label contaminated': 5, 'accuracy - not labeled': 0},
    'high_school_chemistry': {'accuracy - clean': 16, 'accuracy - input contaminated': 0, 'accuracy - input-and-label contaminated': 3, 'accuracy - not labeled': 0},
    'high_school_biology': {'accuracy - clean': 9, 'accuracy - input contaminated': 0, 'accuracy - input-and-label contaminated': 10, 'accuracy - not labeled': 0},
    'middle_school_mathematics': {'accuracy - clean': 15, 'accuracy - input contaminated': 1, 'accuracy - input-and-label contaminated': 3, 'accuracy - not labeled': 0},
    'middle_school_biology': {'accuracy - clean': 10, 'accuracy - input contaminated': 0, 'accuracy - input-and-label contaminated': 11, 'accuracy - not labeled': 0},
    'middle_school_physics': {'accuracy - clean': 7, 'accuracy - input contaminated': 1, 'accuracy - input-and-label contaminated': 11, 'accuracy - not labeled': 0},
    'middle_school_chemistry': {'accuracy - clean': 12, 'accuracy - input contaminated': 0, 'accuracy - input-and-label contaminated': 8, 'accuracy - not labeled': 0},
    'veterinary_medicine': {'accuracy - clean': 13, 'accuracy - input contaminated': 0, 'accuracy - input-and-label contaminated': 10, 'accuracy - not labeled': 0},
    'college_economics': {'accuracy - clean': 19, 'accuracy - input contaminated': 4, 'accuracy - input-and-label contaminated': 32, 'accuracy - not labeled': 0},
    'business_administration': {'accuracy - clean': 13, 'accuracy - input contaminated': 2, 'accuracy - input-and-label contaminated': 18, 'accuracy - not labeled': 0},
    'marxism': {'accuracy - clean': 10, 'accuracy - input contaminated': 1, 'accuracy - input-and-label contaminated': 8, 'accuracy - not labeled': 0},
    'mao_zedong_thought': {'accuracy - clean': 6, 'accuracy - input contaminated': 0, 'accuracy - input-and-label contaminated': 18, 'accuracy - not labeled': 0},
    'education_science': {'accuracy - clean': 11, 'accuracy - input contaminated': 1, 'accuracy - input-and-label contaminated': 17, 'accuracy - not labeled': 0},
    'teacher_qualification': {'accuracy - clean': 18, 'accuracy - input contaminated': 2, 'accuracy - input-and-label contaminated': 23, 'accuracy - not labeled': 1},
    'high_school_politics': {'accuracy - clean': 14, 'accuracy - input contaminated': 2, 'accuracy - input-and-label contaminated': 3, 'accuracy - not labeled': 0},
    'high_school_geography': {'accuracy - clean': 11, 'accuracy - input contaminated': 0, 'accuracy - input-and-label contaminated': 8, 'accuracy - not labeled': 0},
    'middle_school_politics': {'accuracy - clean': 20, 'accuracy - input contaminated': 0, 'accuracy - input-and-label contaminated': 1, 'accuracy - not labeled': 0},
    'middle_school_geography': {'accuracy - clean': 3, 'accuracy - input contaminated': 1, 'accuracy - input-and-label contaminated': 8, 'accuracy - not labeled': 0},
    'modern_chinese_history': {'accuracy - clean': 8, 'accuracy - input contaminated': 0, 'accuracy - input-and-label contaminated': 15, 'accuracy - not labeled': 0},
    'ideological_and_moral_cultivation': {'accuracy - clean': 5, 'accuracy - input contaminated': 0, 'accuracy - input-and-label contaminated': 14, 'accuracy - not labeled': 0},
    'logic': {'accuracy - clean': 15, 'accuracy - input contaminated': 0, 'accuracy - input-and-label contaminated': 7, 'accuracy - not labeled': 0},
    'law': {'accuracy - clean': 15, 'accuracy - input contaminated': 3, 'accuracy - input-and-label contaminated': 6, 'accuracy - not labeled': 0},
    'chinese_language_and_literature': {'accuracy - clean': 13, 'accuracy - input contaminated': 1, 'accuracy - input-and-label contaminated': 9, 'accuracy - not labeled': 0},
    'art_studies': {'accuracy - clean': 14, 'accuracy - input contaminated': 0, 'accuracy - input-and-label contaminated': 19, 'accuracy - not labeled': 0},
    'professional_tour_guide': {'accuracy - clean': 10, 'accuracy - input contaminated': 2, 'accuracy - input-and-label contaminated': 17, 'accuracy - not labeled': 0},
    'legal_professional': {'accuracy - clean': 14, 'accuracy - input contaminated': 2, 'accuracy - input-and-label contaminated': 7, 'accuracy - not labeled': 0},
    'high_school_chinese': {'accuracy - clean': 12, 'accuracy - input contaminated': 0, 'accuracy - input-and-label contaminated': 4, 'accuracy - not labeled': 3},
    'high_school_history': {'accuracy - clean': 12, 'accuracy - input contaminated': 3, 'accuracy - input-and-label contaminated': 5, 'accuracy - not labeled': 0},
    'middle_school_history': {'accuracy - clean': 11, 'accuracy - input contaminated': 1, 'accuracy - input-and-label contaminated': 9, 'accuracy - not labeled': 1},
    'civil_servant': {'accuracy - clean': 19, 'accuracy - input contaminated': 5, 'accuracy - input-and-label contaminated': 17, 'accuracy - not labeled': 6},
    'sports_science': {'accuracy - clean': 8, 'accuracy - input contaminated': 2, 'accuracy - input-and-label contaminated': 9, 'accuracy - not labeled': 0},
    'plant_protection': {'accuracy - clean': 12, 'accuracy - input contaminated': 1, 'accuracy - input-and-label contaminated': 9, 'accuracy - not labeled': 0},
    'basic_medicine': {'accuracy - clean': 9, 'accuracy - input contaminated': 0, 'accuracy - input-and-label contaminated': 10, 'accuracy - not labeled': 0},
    'clinical_medicine': {'accuracy - clean': 14, 'accuracy - input contaminated': 1, 'accuracy - input-and-label contaminated': 7, 'accuracy - not labeled': 0},
    'urban_and_rural_planner': {'accuracy - clean': 28, 'accuracy - input contaminated': 3, 'accuracy - input-and-label contaminated': 15, 'accuracy - not labeled': 0},
    'accountant': {'accuracy - clean': 17, 'accuracy - input contaminated': 7, 'accuracy - input-and-label contaminated': 25, 'accuracy - not labeled': 0},
    'fire_engineer': {'accuracy - clean': 12, 'accuracy - input contaminated': 1, 'accuracy - input-and-label contaminated': 18, 'accuracy - not labeled': 0},
    'environmental_impact_assessment_engineer': {'accuracy - clean': 21, 'accuracy - input contaminated': 2, 'accuracy - input-and-label contaminated': 8, 'accuracy - not labeled': 0},
    'tax_accountant': {'accuracy - clean': 31, 'accuracy - input contaminated': 0, 'accuracy - input-and-label contaminated': 18, 'accuracy - not labeled': 0},
    'physician': {'accuracy - clean': 24, 'accuracy - input contaminated': 1, 'accuracy - input-and-label contaminated': 24, 'accuracy - not labeled': 0},
}

mmlu_category_weights = {
    'business_ethics': {'accuracy - clean': 44, 'accuracy - input contaminated': 16, 'accuracy - input-and-label contaminated': 38, 'accuracy - not labeled': 1},
    'security_studies': {'accuracy - clean': 188, 'accuracy - input contaminated': 9, 'accuracy - input-and-label contaminated': 47, 'accuracy - not labeled': 0},
    'high_school_us_history': {'accuracy - clean': 42, 'accuracy - input contaminated': 0, 'accuracy - input-and-label contaminated': 0, 'accuracy - not labeled': 161},
    'moral_disputes': {'accuracy - clean': 105, 'accuracy - input contaminated': 13, 'accuracy - input-and-label contaminated': 168, 'accuracy - not labeled': 59},
    'philosophy': {'accuracy - clean': 81, 'accuracy - input contaminated': 11, 'accuracy - input-and-label contaminated': 187, 'accuracy - not labeled': 31},
    'public_relations': {'accuracy - clean': 75, 'accuracy - input contaminated': 8, 'accuracy - input-and-label contaminated': 26, 'accuracy - not labeled': 0},
    'high_school_microeconomics': {'accuracy - clean': 82, 'accuracy - input contaminated': 9, 'accuracy - input-and-label contaminated': 146, 'accuracy - not labeled': 0},
    'human_sexuality': {'accuracy - clean': 108, 'accuracy - input contaminated': 3, 'accuracy - input-and-label contaminated': 15, 'accuracy - not labeled': 4},
    'professional_accounting': {'accuracy - clean': 88, 'accuracy - input contaminated': 40, 'accuracy - input-and-label contaminated': 152, 'accuracy - not labeled': 1},
    'high_school_government_and_politics': {'accuracy - clean': 104, 'accuracy - input contaminated': 6, 'accuracy - input-and-label contaminated': 82, 'accuracy - not labeled': 0},
    'sociology': {'accuracy - clean': 105, 'accuracy - input contaminated': 4, 'accuracy - input-and-label contaminated': 91, 'accuracy - not labeled': 0},
    'conceptual_physics': {'accuracy - clean': 79, 'accuracy - input contaminated': 8, 'accuracy - input-and-label contaminated': 147, 'accuracy - not labeled': 0},
    'human_aging': {'accuracy - clean': 208, 'accuracy - input contaminated': 1, 'accuracy - input-and-label contaminated': 13, 'accuracy - not labeled': 0},
    'high_school_psychology': {'accuracy - clean': 108, 'accuracy - input contaminated': 26, 'accuracy - input-and-label contaminated': 162, 'accuracy - not labeled': 248},
    'jurisprudence': {'accuracy - clean': 59, 'accuracy - input contaminated': 5, 'accuracy - input-and-label contaminated': 43, 'accuracy - not labeled': 0},
    'moral_scenarios': {'accuracy - clean': 320, 'accuracy - input contaminated': 0, 'accuracy - input-and-label contaminated': 0, 'accuracy - not labeled': 574},
    'college_medicine': {'accuracy - clean': 107, 'accuracy - input contaminated': 16, 'accuracy - input-and-label contaminated': 44, 'accuracy - not labeled': 5},
    'high_school_world_history': {'accuracy - clean': 61, 'accuracy - input contaminated': 2, 'accuracy - input-and-label contaminated': 0, 'accuracy - not labeled': 173},
    'virology': {'accuracy - clean': 104, 'accuracy - input contaminated': 3, 'accuracy - input-and-label contaminated': 58, 'accuracy - not labeled': 0},
    'high_school_statistics': {'accuracy - clean': 96, 'accuracy - input contaminated': 43, 'accuracy - input-and-label contaminated': 76, 'accuracy - not labeled': 0},
    'nutrition': {'accuracy - clean': 172, 'accuracy - input contaminated': 11, 'accuracy - input-and-label contaminated': 98, 'accuracy - not labeled': 24},
    'abstract_algebra': {'accuracy - clean': 84, 'accuracy - input contaminated': 8, 'accuracy - input-and-label contaminated': 7, 'accuracy - not labeled': 0},
    'high_school_geography': {'accuracy - clean': 91, 'accuracy - input contaminated': 1, 'accuracy - input-and-label contaminated': 105, 'accuracy - not labeled': 0},
    'econometrics': {'accuracy - clean': 62, 'accuracy - input contaminated': 13, 'accuracy - input-and-label contaminated': 38, 'accuracy - not labeled': 0},
    'marketing': {'accuracy - clean': 115, 'accuracy - input contaminated': 15, 'accuracy - input-and-label contaminated': 101, 'accuracy - not labeled': 2},
    'high_school_chemistry': {'accuracy - clean': 108, 'accuracy - input contaminated': 25, 'accuracy - input-and-label contaminated': 69, 'accuracy - not labeled': 0},
    'prehistory': {'accuracy - clean': 154, 'accuracy - input contaminated': 5, 'accuracy - input-and-label contaminated': 107, 'accuracy - not labeled': 57},
    'college_physics': {'accuracy - clean': 25, 'accuracy - input contaminated': 20, 'accuracy - input-and-label contaminated': 57, 'accuracy - not labeled': 0},
    'management': {'accuracy - clean': 35, 'accuracy - input contaminated': 5, 'accuracy - input-and-label contaminated': 62, 'accuracy - not labeled': 0},
    'college_biology': {'accuracy - clean': 91, 'accuracy - input contaminated': 12, 'accuracy - input-and-label contaminated': 40, 'accuracy - not labeled': 0},
    'high_school_biology': {'accuracy - clean': 128, 'accuracy - input contaminated': 17, 'accuracy - input-and-label contaminated': 135, 'accuracy - not labeled': 29},
    'high_school_physics': {'accuracy - clean': 42, 'accuracy - input contaminated': 28, 'accuracy - input-and-label contaminated': 80, 'accuracy - not labeled': 0},
    'logical_fallacies': {'accuracy - clean': 133, 'accuracy - input contaminated': 5, 'accuracy - input-and-label contaminated': 24, 'accuracy - not labeled': 0},
    'medical_genetics': {'accuracy - clean': 49, 'accuracy - input contaminated': 6, 'accuracy - input-and-label contaminated': 43, 'accuracy - not labeled': 1},
    'machine_learning': {'accuracy - clean': 71, 'accuracy - input contaminated': 8, 'accuracy - input-and-label contaminated': 32, 'accuracy - not labeled': 0},
    'professional_law': {'accuracy - clean': 401, 'accuracy - input contaminated': 8, 'accuracy - input-and-label contaminated': 5, 'accuracy - not labeled': 1119},
    'professional_psychology': {'accuracy - clean': 265, 'accuracy - input contaminated': 9, 'accuracy - input-and-label contaminated': 27, 'accuracy - not labeled': 310},
    'global_facts': {'accuracy - clean': 89, 'accuracy - input contaminated': 5, 'accuracy - input-and-label contaminated': 5, 'accuracy - not labeled': 0},
    'us_foreign_policy': {'accuracy - clean': 71, 'accuracy - input contaminated': 3, 'accuracy - input-and-label contaminated': 25, 'accuracy - not labeled': 0},
    'international_law': {'accuracy - clean': 73, 'accuracy - input contaminated': 1, 'accuracy - input-and-label contaminated': 46, 'accuracy - not labeled': 0},
    'clinical_knowledge': {'accuracy - clean': 172, 'accuracy - input contaminated': 6, 'accuracy - input-and-label contaminated': 86, 'accuracy - not labeled': 0},
    'high_school_mathematics': {'accuracy - clean': 178, 'accuracy - input contaminated': 59, 'accuracy - input-and-label contaminated': 32, 'accuracy - not labeled': 0},
    'high_school_computer_science': {'accuracy - clean': 62, 'accuracy - input contaminated': 7, 'accuracy - input-and-label contaminated': 28, 'accuracy - not labeled': 2},
    'college_computer_science': {'accuracy - clean': 68, 'accuracy - input contaminated': 15, 'accuracy - input-and-label contaminated': 15, 'accuracy - not labeled': 1},
    'electrical_engineering': {'accuracy - clean': 75, 'accuracy - input contaminated': 8, 'accuracy - input-and-label contaminated': 61, 'accuracy - not labeled': 0},
    'college_mathematics': {'accuracy - clean': 61, 'accuracy - input contaminated': 13, 'accuracy - input-and-label contaminated': 26, 'accuracy - not labeled': 0},
    'computer_security': {'accuracy - clean': 55, 'accuracy - input contaminated': 8, 'accuracy - input-and-label contaminated': 36, 'accuracy - not labeled': 0},
    'high_school_macroeconomics': {'accuracy - clean': 102, 'accuracy - input contaminated': 14, 'accuracy - input-and-label contaminated': 173, 'accuracy - not labeled': 100},
    'astronomy': {'accuracy - clean': 112, 'accuracy - input contaminated': 4, 'accuracy - input-and-label contaminated': 35, 'accuracy - not labeled': 0},
    'college_chemistry': {'accuracy - clean': 46, 'accuracy - input contaminated': 19, 'accuracy - input-and-label contaminated': 34, 'accuracy - not labeled': 0},
    'high_school_european_history': {'accuracy - clean': 41, 'accuracy - input contaminated': 0, 'accuracy - input-and-label contaminated': 0, 'accuracy - not labeled': 123},
    'miscellaneous': {'accuracy - clean': 256, 'accuracy - input contaminated': 9, 'accuracy - input-and-label contaminated': 40, 'accuracy - not labeled': 477},
    'formal_logic': {'accuracy - clean': 92, 'accuracy - input contaminated': 12, 'accuracy - input-and-label contaminated': 21, 'accuracy - not labeled': 0},
    'elementary_mathematics': {'accuracy - clean': 155, 'accuracy - input contaminated': 31, 'accuracy - input-and-label contaminated': 103, 'accuracy - not labeled': 88},
    'world_religions': {'accuracy - clean': 130, 'accuracy - input contaminated': 4, 'accuracy - input-and-label contaminated': 36, 'accuracy - not labeled': 0},
    'professional_medicine': {'accuracy - clean': 191, 'accuracy - input contaminated': 43, 'accuracy - input-and-label contaminated': 1, 'accuracy - not labeled': 36},
    'anatomy': {'accuracy - clean': 52, 'accuracy - input contaminated': 6, 'accuracy - input-and-label contaminated': 76, 'accuracy - not labeled': 0},
}


ARC_weights = {'accuracy - clean': 836, 'accuracy - input contaminated': 53, 'accuracy - input-and-label contaminated': 283, 'accuracy - not labeled': 0}
hellaswag_weights = {'accuracy - clean': 5169, 'accuracy - input contaminated': 37, 'accuracy - input-and-label contaminated': 673, 'accuracy - not labeled': 4163}

ceval_stem = ['computer_network', 'operating_system', 'computer_architecture', 'college_programming', 'college_physics', 'college_chemistry', 'advanced_mathematics', 'probability_and_statistics', 'discrete_mathematics', 'electrical_engineer', 'metrology_engineer', 'high_school_mathematics', 'high_school_physics', 'high_school_chemistry', 'high_school_biology', 'middle_school_mathematics', 'middle_school_biology', 'middle_school_physics', 'middle_school_chemistry', 'veterinary_medicine']
ceval_social_science = ['college_economics', 'business_administration', 'marxism', 'mao_zedong_thought', 'education_science', 'teacher_qualification', 'high_school_politics', 'high_school_geography', 'middle_school_politics', 'middle_school_geography']
ceval_humanities = ['modern_chinese_history', 'ideological_and_moral_cultivation', 'logic', 'law', 'chinese_language_and_literature', 'art_studies', 'professional_tour_guide', 'legal_professional', 'high_school_chinese', 'high_school_history', 'middle_school_history']
ceval_other = ['civil_servant', 'sports_science', 'plant_protection', 'basic_medicine', 'clinical_medicine', 'urban_and_rural_planner', 'accountant', 'fire_engineer', 'environmental_impact_assessment_engineer', 'tax_accountant', 'physician']
ceval_hard = ['advanced_mathematics', 'discrete_mathematics', 'probability_and_statistics', 'college_chemistry', 'college_physics', 'high_school_mathematics', 'high_school_chemistry', 'high_school_physics']
ceval_all = ceval_stem + ceval_social_science + ceval_humanities + ceval_other

_mmlu_humanities = ['formal_logic', 'high_school_european_history', 'high_school_us_history', 'high_school_world_history', 'international_law', 'jurisprudence', 'logical_fallacies', 'moral_disputes', 'moral_scenarios', 'philosophy', 'prehistory', 'professional_law', 'world_religions']
_mmlu_stem = ['abstract_algebra', 'anatomy', 'astronomy', 'college_biology', 'college_chemistry', 'college_computer_science', 'college_mathematics', 'college_physics', 'computer_security', 'conceptual_physics', 'electrical_engineering', 'elementary_mathematics', 'high_school_biology', 'high_school_chemistry', 'high_school_computer_science', 'high_school_mathematics', 'high_school_physics', 'high_school_statistics', 'machine_learning']
_mmlu_social_science = ['econometrics', 'high_school_geography', 'high_school_government_and_politics', 'high_school_macroeconomics', 'high_school_microeconomics', 'high_school_psychology', 'human_sexuality', 'professional_psychology', 'public_relations', 'security_studies', 'sociology', 'us_foreign_policy']
_mmlu_other = ['business_ethics', 'clinical_knowledge', 'college_medicine', 'global_facts', 'human_aging', 'management', 'marketing', 'medical_genetics', 'miscellaneous', 'nutrition', 'professional_accounting', 'professional_medicine', 'virology']
_mmlu_all = _mmlu_humanities + _mmlu_stem + _mmlu_social_science + _mmlu_other

ceval_name_and_subsets = [
    ('ceval', ceval_all),
    ('ceval-stem', ceval_stem),
    ('ceval-social-science', ceval_social_science),
    ('ceval-humanities', ceval_humanities),
    ('ceval-other', ceval_other),
    ('ceval-hard', ceval_hard)
]

mmlu_name_and_subsets = [
    ('mmlu', _mmlu_all),
    ('mmlu-humanities', _mmlu_humanities),
    ('mmlu-stem', _mmlu_stem),
    ('mmlu-social-science', _mmlu_social_science),
    ('mmlu-other', _mmlu_other)
]

summary_groups = []
for metric_name in ['accuracy - clean', 'accuracy - input contaminated', 'accuracy - input-and-label contaminated']:
    for dataset_abbr, subsets in ceval_name_and_subsets:
        weights = {f'ceval-{i}': ceval_category_weights[i][metric_name] for i in subsets}
        subsets = [[f'ceval-{i}', metric_name] for i in subsets]
        summary_groups.append(
            {
                'name': dataset_abbr,
                'subsets': subsets,
                'metric': metric_name,
                'weights': weights,
            }
        )

    for dataset_abbr, subsets in mmlu_name_and_subsets:
        weights = {f'lukaemon_mmlu_{i}': mmlu_category_weights[i][metric_name] for i in subsets}
        subsets = [[f'lukaemon_mmlu_{i}', metric_name] for i in subsets]
        summary_groups.append(
            {
                'name': dataset_abbr,
                'subsets': subsets,
                'metric': metric_name,
                'weights': weights,
            }
        )

    summary_groups.append(
        {
            'name': 'hellaswag',
            'subsets': [['hellaswag', metric_name]],
            'metric': metric_name,
            'weights': {'hellaswag': hellaswag_weights[metric_name]}
        }
    )

    summary_groups.append(
        {
            'name': 'ARC-c-test',
            'subsets': [['ARC-c-test', metric_name]],
            'metric': metric_name,
            'weights': {'ARC-c-test': ARC_weights[metric_name]}
        }
    )

summarizer = dict(
    type=CircularSummarizer,
    metric_types=['accuracy - clean', 'accuracy - input contaminated', 'accuracy - input-and-label contaminated'],
    dataset_abbrs = ['ceval', 'ceval-stem', 'ceval-social-science', 'ceval-humanities', 'ceval-other', 'ceval-hard', 'mmlu', 'mmlu-humanities', 'mmlu-stem', 'mmlu-social-science', 'mmlu-other', 'hellaswag', 'ARC-c-test'],
    summary_groups=summary_groups,
)
