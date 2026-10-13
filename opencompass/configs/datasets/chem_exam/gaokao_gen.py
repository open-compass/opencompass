from opencompass.models.openai_api import OpenAISDK
from opencompass.evaluator import GenericLLMEvaluator
from opencompass.openicl.icl_prompt_template import PromptTemplate
from opencompass.openicl.icl_retriever import ZeroRetriever
from opencompass.openicl.icl_inferencer import GenInferencer
from opencompass.datasets import ChemExamDataset
from opencompass.datasets.chem_exam import chem_exam_score_llmjudge_postprocess

chem_gaokao_reader_cfg = dict(
    input_columns=['prompt'],
    output_column='output'
)

chem_gaokao_hint = 'Answer the following chemistry question. Please reason step by step, and put your final answer within \\boxed{}.'

chem_gaokao_infer_cfg = dict(
    prompt_template=dict(
        type=PromptTemplate,
        template=dict(
            round=[
                dict(
                    role='HUMAN',
                    prompt=f'{chem_gaokao_hint}\n\nQuestion: {{prompt}}\nAnswer: ',
                ),
            ],
        ),
    ),
    retriever=dict(type=ZeroRetriever),
    inferencer=dict(type=GenInferencer),
)

verify_prompt_yes_no = "Below is a chemistry exam question and a student's answer:\n##Question##\n{prompt}\n\n##Student's Answer##\n{prediction}\n\nThe standard answer for this question is as follows:\n##Standard Answer##\n{output}\n\nNow, based on the standard answer, determine whether the student's answer is correct. (Please note that the same chemical expression may have different formats or equivalent forms). You only need to focus on:\n1. Whether the student's answer matches the result of the standard answer (without focusing too much on the method).\n2. Whether the student's answer seems to be guessed or is a vague answer. If the student's answer is correct (if there are multiple questions, all sub-questions must be answered correctly), please reply directly with:\n**Correct Answer**\nIf the student's answer is incorrect, please reply directly with:\n**Incorrect Answer**"

verify_prompt_score = """Below is a chemistry exam question and a student's answer:

##Question##
{prompt}

##Student's Answer##
{prediction}

##Standard Answer##
{output}

Now, please compare the student's answer with the standard answer. Assume the question consists of multiple sub-questions. For each sub-question, determine if the student's answer is correct by the following criteria:

Evaluation criteria:
1. Only consider whether the final result of each sub-question matches the standard answer. Equivalent chemical expressions or formats should be accepted.
2. Do not focus on the student's method, only the correctness of the final result.
3. If the correct answer is a chemical formula and the student provides a description instead, the description must be specific and fully correspond to the chemical formula. Vague or imprecise descriptions are incorrect.
4. If a student's answer is vague, unclear, or appears to be guessed, mark it as incorrect.
5. If a sub-question contains multiple parts or answers, award partial credit based on how many parts of the answer are correct. Each correct part within a sub-question should be given partial credit.

Return a single score: the proportion of correctly answered sub-questions (number of correct answers (might be float number) divided by the total number of sub-questions).

Format your final answer as: \\boxed{{score}}, where score is a decimal between 0 and 1."""

api_meta_template = dict(
    round=[
        dict(role='HUMAN', api_role='HUMAN'),
        dict(role='BOT', api_role='BOT', generate=True),
    ]
)

chem_gaokao_eval_cfg = dict(
    evaluator=dict(
        type=GenericLLMEvaluator,
        prompt_template=dict(
            type=PromptTemplate,
            template=dict(
                round=[
                    dict(role='HUMAN', prompt=verify_prompt_score),
                ]
            ),
        ),
        dataset_cfg=dict(
            type=ChemExamDataset,
            path='opencompass/Chem_exam_gaokao',
            reader_cfg=chem_gaokao_reader_cfg,
        ),
        judge_cfg=dict(),
        dict_postprocessor=dict(type=chem_exam_score_llmjudge_postprocess),
    ),
)

chem_gaokao_instruct_datasets = [
    dict(
        abbr=f'Chem_exam-gaokao',
        type=ChemExamDataset,
        path='opencompass/Chem_exam_gaokao',
        reader_cfg=chem_gaokao_reader_cfg,
        infer_cfg=chem_gaokao_infer_cfg,
        eval_cfg=chem_gaokao_eval_cfg,
    )
]
