from opencompass.openicl.icl_prompt_template import PromptTemplate
from opencompass.openicl.icl_retriever import FixKRetriever
from opencompass.openicl.icl_inferencer import GenInferencer
from opencompass.openicl.icl_evaluator import AccEvaluator
from opencompass.datasets import CEvalDataset
from opencompass.utils.text_postprocessors import first_capital_postprocess


ceval_subject_mapping = {
    'computer_network': ['Computer Network', '计算机网络', 'STEM'],
    'operating_system': ['Operating System', '操作系统', 'STEM'],
    'computer_architecture': ['Computer Architecture', '计算机组成', 'STEM'],
    'college_programming': ['College Programming', '大学编程', 'STEM'],
    'college_physics': ['College Physics', '大学物理', 'STEM'],
    'college_chemistry': ['College Chemistry', '大学化学', 'STEM'],
    'advanced_mathematics': ['Advanced Mathematics', '高等数学', 'STEM'],
    'probability_and_statistics': ['Probability and Statistics', '概率统计', 'STEM'],
    'discrete_mathematics': ['Discrete Mathematics', '离散数学', 'STEM'],
    'electrical_engineer': ['Electrical Engineer', '注册电气工程师', 'STEM'],
    'metrology_engineer': ['Metrology Engineer', '注册计量师', 'STEM'],
    'high_school_mathematics': ['High School Mathematics', '高中数学', 'STEM'],
    'high_school_physics': ['High School Physics', '高中物理', 'STEM'],
    'high_school_chemistry': ['High School Chemistry', '高中化学', 'STEM'],
    'high_school_biology': ['High School Biology', '高中生物', 'STEM'],
    'middle_school_mathematics': ['Middle School Mathematics', '初中数学', 'STEM'],
    'middle_school_biology': ['Middle School Biology', '初中生物', 'STEM'],
    'middle_school_physics': ['Middle School Physics', '初中物理', 'STEM'],
    'middle_school_chemistry': ['Middle School Chemistry', '初中化学', 'STEM'],
    'veterinary_medicine': ['Veterinary Medicine', '兽医学', 'STEM'],
    'college_economics': ['College Economics', '大学经济学', 'Social Science'],
    'business_administration': ['Business Administration', '工商管理', 'Social Science'],
    'marxism': ['Marxism', '马克思主义基本原理', 'Social Science'],
    'mao_zedong_thought': ['Mao Zedong Thought', '毛泽东思想和中国特色社会主义理论体系概论', 'Social Science'],
    'education_science': ['Education Science', '教育学', 'Social Science'],
    'teacher_qualification': ['Teacher Qualification', '教师资格', 'Social Science'],
    'high_school_politics': ['High School Politics', '高中政治', 'Social Science'],
    'high_school_geography': ['High School Geography', '高中地理', 'Social Science'],
    'middle_school_politics': ['Middle School Politics', '初中政治', 'Social Science'],
    'middle_school_geography': ['Middle School Geography', '初中地理', 'Social Science'],
    'modern_chinese_history': ['Modern Chinese History', '近代史纲要', 'Humanities'],
    'ideological_and_moral_cultivation': ['Ideological and Moral Cultivation', '思想道德修养与法律基础', 'Humanities'],
    'logic': ['Logic', '逻辑学', 'Humanities'],
    'law': ['Law', '法学', 'Humanities'],
    'chinese_language_and_literature': ['Chinese Language and Literature', '中国语言文学', 'Humanities'],
    'art_studies': ['Art Studies', '艺术学', 'Humanities'],
    'professional_tour_guide': ['Professional Tour Guide', '导游资格', 'Humanities'],
    'legal_professional': ['Legal Professional', '法律职业资格', 'Humanities'],
    'high_school_chinese': ['High School Chinese', '高中语文', 'Humanities'],
    'high_school_history': ['High School History', '高中历史', 'Humanities'],
    'middle_school_history': ['Middle School History', '初中历史', 'Humanities'],
    'civil_servant': ['Civil Servant', '公务员', 'Other'],
    'sports_science': ['Sports Science', '体育学', 'Other'],
    'plant_protection': ['Plant Protection', '植物保护', 'Other'],
    'basic_medicine': ['Basic Medicine', '基础医学', 'Other'],
    'clinical_medicine': ['Clinical Medicine', '临床医学', 'Other'],
    'urban_and_rural_planner': ['Urban and Rural Planner', '注册城乡规划师', 'Other'],
    'accountant': ['Accountant', '注册会计师', 'Other'],
    'fire_engineer': ['Fire Engineer', '注册消防工程师', 'Other'],
    'environmental_impact_assessment_engineer': ['Environmental Impact Assessment Engineer', '环境影响评价工程师', 'Other'],
    'tax_accountant': ['Tax Accountant', '税务师', 'Other'],
    'physician': ['Physician', '医师资格', 'Other'],
}
ceval_all_sets = list(ceval_subject_mapping.keys())

ceval_datasets = []
for _split in ['val']:
    for _name in ceval_all_sets:
        _ch_name = ceval_subject_mapping[_name][1]
        ceval_infer_cfg = dict(
            ice_template=dict(
                type=PromptTemplate,
                template=dict(
                    begin='</E>',
                    round=[
                        dict(
                            role='HUMAN',
                            prompt=
                            f'以下是中国关于{_ch_name}考试的单项选择题，请选出其中的正确答案。\n{{question}}\nA. {{A}}\nB. {{B}}\nC. {{C}}\nD. {{D}}\n答案: '
                        ),
                        dict(role='BOT', prompt='{answer}'),
                    ]),
                ice_token='</E>',
            ),
            retriever=dict(type=FixKRetriever, fix_id_list=[0, 1, 2, 3, 4]),
            inferencer=dict(type=GenInferencer),
        )

        ceval_eval_cfg = dict(
            evaluator=dict(type=AccEvaluator),
            pred_postprocessor=dict(type=first_capital_postprocess))

        ceval_datasets.append(
            dict(
                type=CEvalDataset,
                path='opencompass/ceval-exam',
                name=_name,
                abbr='ceval-' + _name if _split == 'val' else 'ceval-test-' +
                _name,
                reader_cfg=dict(
                    input_columns=['question', 'A', 'B', 'C', 'D'],
                    output_column='answer',
                    train_split='dev',
                    test_split=_split),
                infer_cfg=ceval_infer_cfg,
                eval_cfg=ceval_eval_cfg,
            ))

del _split, _name, _ch_name
