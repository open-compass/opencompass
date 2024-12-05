from opencompass.openicl.icl_prompt_template import PromptTemplate
from opencompass.openicl.icl_retriever import ZeroRetriever
from opencompass.openicl.icl_inferencer import GenInferencer
from opencompass.openicl.icl_evaluator import AccEvaluator
from opencompass.datasets import CMMLUDataset
from opencompass.utils.text_postprocessors import match_answer_pattern
from opencompass.openicl.icl_evaluator import LMEvaluator
from opencompass.datasets import generic_llmjudge_postprocess

cmmlu_subject_mapping = {
    'agronomy': '农学',
    'anatomy': '解剖学',
    'ancient_chinese': '古汉语',
    'arts': '艺术学',
    'astronomy': '天文学',
    'business_ethics': '商业伦理',
    'chinese_civil_service_exam': '中国公务员考试',
    'chinese_driving_rule': '中国驾驶规则',
    'chinese_food_culture': '中国饮食文化',
    'chinese_foreign_policy': '中国外交政策',
    'chinese_history': '中国历史',
    'chinese_literature': '中国文学',
    'chinese_teacher_qualification': '中国教师资格',
    'clinical_knowledge': '临床知识',
    'college_actuarial_science': '大学精算学',
    'college_education': '大学教育学',
    'college_engineering_hydrology': '大学工程水文学',
    'college_law': '大学法律',
    'college_mathematics': '大学数学',
    'college_medical_statistics': '大学医学统计',
    'college_medicine': '大学医学',
    'computer_science': '计算机科学',
    'computer_security': '计算机安全',
    'conceptual_physics': '概念物理学',
    'construction_project_management': '建设工程管理',
    'economics': '经济学',
    'education': '教育学',
    'electrical_engineering': '电气工程',
    'elementary_chinese': '小学语文',
    'elementary_commonsense': '小学常识',
    'elementary_information_and_technology': '小学信息技术',
    'elementary_mathematics': '初等数学',
    'ethnology': '民族学',
    'food_science': '食品科学',
    'genetics': '遗传学',
    'global_facts': '全球事实',
    'high_school_biology': '高中生物',
    'high_school_chemistry': '高中化学',
    'high_school_geography': '高中地理',
    'high_school_mathematics': '高中数学',
    'high_school_physics': '高中物理学',
    'high_school_politics': '高中政治',
    'human_sexuality': '人类性行为',
    'international_law': '国际法学',
    'journalism': '新闻学',
    'jurisprudence': '法理学',
    'legal_and_moral_basis': '法律与道德基础',
    'logical': '逻辑学',
    'machine_learning': '机器学习',
    'management': '管理学',
    'marketing': '市场营销',
    'marxist_theory': '马克思主义理论',
    'modern_chinese': '现代汉语',
    'nutrition': '营养学',
    'philosophy': '哲学',
    'professional_accounting': '专业会计',
    'professional_law': '专业法学',
    'professional_medicine': '专业医学',
    'professional_psychology': '专业心理学',
    'public_relations': '公共关系',
    'security_study': '安全研究',
    'sociology': '社会学',
    'sports_science': '体育学',
    'traditional_chinese_medicine': '中医中药',
    'virology': '病毒学',
    'world_history': '世界历史',
    'world_religions': '世界宗教'
}

QUERY_TEMPLATE = """
你回答的最后一行**必须**是以下格式 '答案: $选项' (不带引号), 其中选项是ABCD之一.

{question}

A) {A}
B) {B}
C) {C}
D) {D}
""".strip()



GRADER_TEMPLATE = """
    Please as a grading expert, judge whether the final answers given by the candidates below are consistent with the standard answers, that is, whether the candidates answered correctly. 
    
    Here are some evaluation criteria:
    1. Please refer to the given standard answer. You don't need to re-generate the answer to the question because the standard answer has been given. You only need to judge whether the candidate's answer is consistent with the standard answer according to the form of the question. Don't try to answer the original question. You can assume that the standard answer is definitely correct.
    2. Because the candidate's answer may be different from the standard answer in the form of expression, before making a judgment, please understand the question and the standard answer first, and then judge whether the candidate's answer is correct, but be careful not to try to answer the original question.
    3. Some answers may contain multiple items, such as multiple-choice questions, multiple-select questions, fill-in-the-blank questions, etc. As long as the answer is the same as the standard answer, it is enough. For multiple-select questions and multiple-blank fill-in-the-blank questions, the candidate needs to answer all the corresponding options or blanks correctly to be considered correct.
    4. Some answers may be expressed in different ways, such as some answers may be a mathematical expression, some answers may be a textual description, as long as the meaning expressed is the same. And some formulas are expressed in different ways, but they are equivalent and correct.

    Please judge whether the following answers are consistent with the standard answer based on the above criteria. Grade the predicted answer of this new question as one of:
    A: CORRECT 
    B: INCORRECT
    Just return the letters "A" or "B", with no text around it.

    Here is your task. Simply reply with either CORRECT, INCORRECT. Don't apologize or correct yourself if there was a mistake; we are just trying to grade the answer.

    <Original Question Begin>: \n {question}\n A) {A}\n B) {B}\n C) {C}\n D) {D}\n<Original Question End>\n\n
    <Gold Target Begin>: \n{answer}\n<Gold Target End>\n\n
    <Predicted Answer Begin>: \n{prediction}\n<Predicted End>\n\n
    Judging the correctness of candidates' answers:
""".strip()

cmmlu_all_sets = list(cmmlu_subject_mapping.keys())

cmmlu_datasets = []
for _name in cmmlu_all_sets:
    _ch_name = cmmlu_subject_mapping[_name]
    prompt_prefix = f'请回答以下关于{_ch_name}的单项选择题, '
    cmmlu_infer_cfg = dict(
        prompt_template=dict(
            type=PromptTemplate,
            template=dict(
                round=[
                    dict(role='HUMAN', prompt=prompt_prefix+QUERY_TEMPLATE),
                ],
            ),
        ),
        retriever=dict(type=ZeroRetriever),
        inferencer=dict(type=GenInferencer),
    )

    cmmlu_eval_cfg = dict(
        evaluator=dict(
            type=LMEvaluator,
            prompt_template=dict(
                type=PromptTemplate,
                template=dict(
                begin=[
                    dict(
                        role='SYSTEM',
                        fallback_role='HUMAN',
                        prompt="You are a helpful assistant who evaluates the correctness and quality of models' outputs.")
                ],
                    round=[
                    dict(
                        role='HUMAN',
                        prompt = GRADER_TEMPLATE
                    ),
                ]),
            ),
            dict_postprocessor=dict(type=generic_llmjudge_postprocess),
        ),
        pred_role='BOT',
    )
    cmmlu_datasets.append(
        dict(
            type=CMMLUDataset,
            path='opencompass/cmmlu',
            name=_name,
            abbr=f'cmmlu-{_name}',
            reader_cfg=dict(
                input_columns=['question', 'A', 'B', 'C', 'D'],
                output_column='answer',
                train_split='dev',
                test_split='test'),
            infer_cfg=cmmlu_infer_cfg,
            eval_cfg=cmmlu_eval_cfg,
            mode='singlescore',
        ))

del _name, _ch_name
