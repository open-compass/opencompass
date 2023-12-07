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


ceval_stem = ['computer_network', 'operating_system', 'computer_architecture', 'college_programming', 'college_physics', 'college_chemistry', 'advanced_mathematics', 'probability_and_statistics', 'discrete_mathematics', 'electrical_engineer', 'metrology_engineer', 'high_school_mathematics', 'high_school_physics', 'high_school_chemistry', 'high_school_biology', 'middle_school_mathematics', 'middle_school_biology', 'middle_school_physics', 'middle_school_chemistry', 'veterinary_medicine']
ceval_social_science = ['college_economics', 'business_administration', 'marxism', 'mao_zedong_thought', 'education_science', 'teacher_qualification', 'high_school_politics', 'high_school_geography', 'middle_school_politics', 'middle_school_geography']
ceval_humanities = ['modern_chinese_history', 'ideological_and_moral_cultivation', 'logic', 'law', 'chinese_language_and_literature', 'art_studies', 'professional_tour_guide', 'legal_professional', 'high_school_chinese', 'high_school_history', 'middle_school_history']
ceval_other = ['civil_servant', 'sports_science', 'plant_protection', 'basic_medicine', 'clinical_medicine', 'urban_and_rural_planner', 'accountant', 'fire_engineer', 'environmental_impact_assessment_engineer', 'tax_accountant', 'physician']
ceval_hard = ['advanced_mathematics', 'discrete_mathematics', 'probability_and_statistics', 'college_chemistry', 'college_physics', 'high_school_mathematics', 'high_school_chemistry', 'high_school_physics']
ceval_all = ceval_stem + ceval_social_science + ceval_humanities + ceval_other

name_and_subsets = [
    ('ceval', ceval_all),
    ('ceval-stem', ceval_stem),
    ('ceval-social-science', ceval_social_science),
    ('ceval-humanities', ceval_humanities),
    ('ceval-other', ceval_other),
    ('ceval-hard', ceval_hard)
]

ceval_summary_groups = []
for metric_name in ['accuracy - clean', 'accuracy - input contaminated', 'accuracy - input-and-label contaminated']:
    for dataset_abbr, subsets in name_and_subsets:
        weights = {f'ceval-{i}': ceval_category_weights[i][metric_name] for i in subsets}
        subsets = [[f'ceval-{i}', metric_name] for i in subsets]
        ceval_summary_groups.append(
            {
                'name': dataset_abbr,
                'subsets': subsets,
                'metric': metric_name,
                'weights': weights,
            }
        )

ceval_summarizer = dict(
    type=CircularSummarizer,
    metric_types=['accuracy - clean', 'accuracy - input contaminated', 'accuracy - input-and-label contaminated'],
    dataset_abbrs = [
        'ceval-computer_network',
        'ceval-operating_system',
        'ceval-computer_architecture',
        'ceval-college_programming',
        'ceval-college_physics',
        'ceval-college_chemistry',
        'ceval-advanced_mathematics',
        'ceval-probability_and_statistics',
        'ceval-discrete_mathematics',
        'ceval-electrical_engineer',
        'ceval-metrology_engineer',
        'ceval-high_school_mathematics',
        'ceval-high_school_physics',
        'ceval-high_school_chemistry',
        'ceval-high_school_biology',
        'ceval-middle_school_mathematics',
        'ceval-middle_school_biology',
        'ceval-middle_school_physics',
        'ceval-middle_school_chemistry',
        'ceval-veterinary_medicine',
        'ceval-college_economics',
        'ceval-business_administration',
        'ceval-marxism',
        'ceval-mao_zedong_thought',
        'ceval-education_science',
        'ceval-teacher_qualification',
        'ceval-high_school_politics',
        'ceval-high_school_geography',
        'ceval-middle_school_politics',
        'ceval-middle_school_geography',
        'ceval-modern_chinese_history',
        'ceval-ideological_and_moral_cultivation',
        'ceval-logic',
        'ceval-law',
        'ceval-chinese_language_and_literature',
        'ceval-art_studies',
        'ceval-professional_tour_guide',
        'ceval-legal_professional',
        'ceval-high_school_chinese',
        'ceval-high_school_history',
        'ceval-middle_school_history',
        'ceval-civil_servant',
        'ceval-sports_science',
        'ceval-plant_protection',
        'ceval-basic_medicine',
        'ceval-clinical_medicine',
        'ceval-urban_and_rural_planner',
        'ceval-accountant',
        'ceval-fire_engineer',
        'ceval-environmental_impact_assessment_engineer',
        'ceval-tax_accountant',
        'ceval-physician',
        'ceval-humanities',
        'ceval-stem',
        'ceval-social-science',
        'ceval-other',
        'ceval-hard',
        'ceval',
    ],
    summary_groups=ceval_summary_groups,
)
