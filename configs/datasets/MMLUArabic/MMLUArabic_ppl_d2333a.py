from opencompass.openicl.icl_prompt_template import PromptTemplate
from opencompass.openicl.icl_retriever import FixKRetriever
from opencompass.openicl.icl_inferencer import PPLInferencer
from opencompass.openicl.icl_evaluator import AccEvaluator
from opencompass.datasets import MMLUArabicDataset

# None of the MMLUArabic dataset in huggingface is correctly parsed, so we use our own dataset reader
# Please download the dataset from https://people.eecs.berkeley.edu/~hendrycks/data.tar

MMLUArabic_reader_cfg = dict(
    input_columns=['input', 'A', 'B', 'C', 'D'],
    output_column='target',
    train_split='dev')

MMLUArabic_all_sets = ['abstract_algebra', 'anatomy', 'astronomy', 'business_ethics', 'clinical_knowledge', 'college_biology', 'college_chemistry', 'college_computer_science', 'college_mathematics', 'college_medicine', 'college_physics', 'computer_security', 'conceptual_physics', 'econometrics', 'electrical_engineering', 'elementary_mathematics', 'formal_logic', 'global_facts', 'high_school_biology', 'high_school_chemistry', 'high_school_computer_science', 'high_school_european_history', 'high_school_geography', 'high_school_government_and_politics', 'high_school_macroeconomics', 'high_school_mathematics', 'high_school_microeconomics', 'high_school_physics', 'high_school_psychology', 'high_school_statistics', 'high_school_us_history', 'high_school_world_history', 'human_aging', 'human_sexuality', 'international_law', 'jurisprudence', 'logical_fallacies', 'machine_learning', 'management', 'marketing', 'medical_genetics', 'miscellaneous', 'moral_disputes', 'moral_scenarios', 'nutrition', 'philosophy', 'prehistory', 'professional_accounting', 'professional_law', 'professional_medicine', 'professional_psychology', 'public_relations', 'security_studies', 'sociology', 'us_foreign_policy', 'virology', 'world_religions']
MMLUArabic_all_sets_ar = ['جبر_تجريدي', 'تشريح', 'علم_الفلك', 'أخلاقيات_الأعمال', 'المعرفة_السريرية', 'علم_الأحياء_الجامعي', 'كيمياء_جامعية', 'علوم_الحاسوب_الجامعية', 'رياضيات_جامعية', 'طب_جامعي', 'فيزياء_جامعية', 'أمان_الحاسوب', 'فيزياء_مفاهيمية', 'الاقتصاد_القياسي', 'هندسة_كهربائية', 'رياضيات_ابتدائية', 'منطق_رسمي', 'حقائق_عالمية', 'علم_الأحياء_الثانوي', 'كيمياء_ثانوية', 'علوم_الحاسوب_الثانوية', 'تاريخ_أوروبا_الثانوي', 'جغرافية_ثانوية', 'الحكومة_والسياسة_الثانوية', 'اقتصاد_كلي_ثانوي', 'رياضيات_ثانوية', 'اقتصاد_جزئي_ثانوي', 'فيزياء_ثانوية', 'علم_النفس_الثانوي', 'إحصاء_ثانوي', 'تاريخ_الولايات_المتحدة_الثانوي', 'تاريخ_العالم_الثانوي', 'شيخوخة_الإنسان', 'جنسانية_بشرية', 'قانون_دولي', 'فقه', 'أخطاء_منطقية', 'تعلم_الآلة', 'إدارة', 'تسويق', 'جينات_طبية', 'متفرقات', 'نزاعات_أخلاقية', 'سيناريوهات_أخلاقية', 'تغذية', 'فلسفة', 'ما_قبل_التاريخ', 'محاسبة_مهنية', 'قانون_مهني', 'طب_مهني', 'علم_النفس_المهني', 'علاقات_عامة', 'دراسات_الأمان', 'علم_الاجتماع', 'سياسة_خارجية_أمريكية', 'علم_الفيروسات', 'أديان_العالم']

MMLUArabic_datasets = []
for _name, _name_ar in zip(MMLUArabic_all_sets, MMLUArabic_all_sets_ar):
    # _hint = f'The following are multiple choice questions (with answers) about  {_name.replace("_", " ")}.\n\n'
    _hint = f"فيما يلي أسئلة الاختيار من متعدد حول {' '.join(_name_ar.split('_'))}\n\n"
    # question_overall = '{input}\nA. {A}\nB. {B}\nC. {C}\nD. {D}'
    question_overall = 'سؤال: {input}\nA. {A}\nB. {B}\nC. {C}\nD. {D}'
    MMLUArabic_infer_cfg = dict(
        ice_template=dict(
            type=PromptTemplate,
            template={opt: f'{question_overall}\nإجابة: {opt}\n' for opt in ['A', 'B', 'C', 'D']},
        ),
        prompt_template=dict(
            type=PromptTemplate,
            template={opt: f'{_hint}</E>{question_overall}\nإجابة: {opt}' for opt in ['A', 'B', 'C', 'D']},
            ice_token='</E>',
        ),
        retriever=dict(type=FixKRetriever, fix_id_list=[0, 1, 2, 3, 4]),
        inferencer=dict(type=PPLInferencer),
    )

    MMLUArabic_eval_cfg = dict(evaluator=dict(type=AccEvaluator), )

    MMLUArabic_datasets.append(
        dict(
            abbr=f'acegpt_MMLUArabic_{_name}',
            type=MMLUArabicDataset,
            path='./data/MMLUArabic/',
            name=_name,
            reader_cfg=MMLUArabic_reader_cfg,
            infer_cfg=MMLUArabic_infer_cfg,
            eval_cfg=MMLUArabic_eval_cfg,
        ))

del _name, _hint
