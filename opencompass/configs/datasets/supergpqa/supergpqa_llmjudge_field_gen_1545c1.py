from opencompass.datasets.supergpqa.supergpqa import (
    SuperGPQADataset,
    supergpqa_llmjudge_postprocess,
)
from opencompass.openicl.icl_inferencer import GenInferencer
from opencompass.openicl.icl_prompt_template import PromptTemplate
from opencompass.openicl.icl_retriever import ZeroRetriever
from opencompass.evaluator import GenericLLMEvaluator

field_list = [
    'Electronic Science and Technology',
    'Philosophy',
    'Traditional Chinese Medicine',
    'Applied Economics',
    'Mathematics',
    'Physics',
    'Clinical Medicine',
    'Computer Science and Technology',
    'Information and Communication Engineering',
    'Control Science and Engineering',
    'Theoretical Economics',
    'Law',
    'History',
    'Basic Medicine',
    'Education',
    'Materials Science and Engineering',
    'Electrical Engineering',
    'Systems Science',
    'Power Engineering and Engineering Thermophysics',
    'Military Science',
    'Biology',
    'Business Administration',
    'Language and Literature',
    'Public Health and Preventive Medicine',
    'Political Science',
    'Chemistry',
    'Hydraulic Engineering',
    'Chemical Engineering and Technology',
    'Pharmacy',
    'Geography',
    'Art Studies',
    'Architecture',
    'Forestry Engineering',
    'Public Administration',
    'Oceanography',
    'Journalism and Communication',
    'Nuclear Science and Technology',
    'Weapon Science and Technology',
    'Naval Architecture and Ocean Engineering',
    'Environmental Science and Engineering',
    'Transportation Engineering',
    'Geology',
    'Physical Oceanography',
    'Musicology',
    'Stomatology',
    'Aquaculture',
    'Mechanical Engineering',
    'Aeronautical and Astronautical Science and Technology',
    'Civil Engineering',
    'Mechanics',
    'Petroleum and Natural Gas Engineering',
    'Sociology',
    'Food Science and Engineering',
    'Agricultural Engineering',
    'Surveying and Mapping Science and Technology',
    'Metallurgical Engineering',
    'Library, Information and Archival Management',
    'Mining Engineering',
    'Astronomy',
    'Geological Resources and Geological Engineering',
    'Atmospheric Science',
    'Optical Engineering',
    'Animal Husbandry',
    'Geophysics',
    'Crop Science',
    'Management Science and Engineering',
    'Psychology',
    'Forestry',
    'Textile Science and Engineering',
    'Veterinary Medicine',
    'Instrument Science and Technology',
    'Physical Education',
]

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

    <Original Question Begin>: {infer_prompt}\n<Original Question End>\n\n
    <Gold Target Begin>: \n{answer_letter}\n<Gold Target End>\n\n
    <Predicted Answer Begin>: \n{prediction}\n<Predicted End>\n\n
    Judging the correctness of candidates' answers:
""".strip()

# Reader configuration
reader_cfg = dict(
    input_columns=[
        'question',
        'options',
        'discipline',
        'field',
        'subfield',
        'difficulty',
        'infer_prompt',
        'prompt_mode',
    ],
    output_column='answer_letter',
)

# Inference configuration
infer_cfg = dict(
    prompt_template=dict(
        type=PromptTemplate,
        template=dict(
            round=[
                dict(
                    role='HUMAN',
                    prompt='{infer_prompt}',
                ),
            ],
        ),
    ),
    retriever=dict(type=ZeroRetriever),
    inferencer=dict(type=GenInferencer),
)
supergpqa_datasets = []
for field in field_list:
    supergpqa_datasets.append(
        dict(
            type=SuperGPQADataset,
            abbr=f'supergpqa_{field.replace(" ", "_")}',
            field=field,
            path='m-a-p/SuperGPQA',
            prompt_mode='zero-shot',
            reader_cfg=reader_cfg,
            infer_cfg=infer_cfg,
            eval_cfg=dict(
                evaluator=dict(
                    type=GenericLLMEvaluator,
                    prompt_template=dict(
                        type=PromptTemplate,
                        template=dict(
                            begin=[
                                dict(
                                    role='SYSTEM',
                                    fallback_role='HUMAN',
                                    prompt="You are a helpful assistant who evaluates the correctness and quality of models' outputs.",
                                )
                            ],
                            round=[
                                dict(role='HUMAN', prompt=GRADER_TEMPLATE),
                            ],
                        ),
                    ),
                    dataset_cfg=dict(
                        type=SuperGPQADataset,
                        field=field,
                        path='m-a-p/SuperGPQA',
                        prompt_mode='zero-shot',
                        reader_cfg=reader_cfg,
                    ),
                    judge_cfg=dict(),
                    dict_postprocessor=dict(
                        type=supergpqa_llmjudge_postprocess
                    ),
                ),
            ),
        )
    )
