from opencompass.datasets import EESEDataset
from opencompass.openicl.icl_prompt_template import PromptTemplate
from opencompass.openicl.icl_retriever import ZeroRetriever
from opencompass.openicl.icl_inferencer import GenInferencer
from opencompass.evaluator import GenericLLMEvaluator
from opencompass.datasets.eese.eese_postprocessors import eese_score_postprocess_dict

# ----------------------------- Detailed Config -----------------------------


# Construct the prompts for tested models

ANSWER_TEMPLATE = """
Question:{question}\n

Question Type:{question_type}
if the question type is closed-ended, please answer the question directly(if it's a single/multiple-choice question, only provide the corresponding letters of your answer options). Please do not provide any analysis process.
if the question type is open-ended, please provide the problem-solving process.
""".strip()

eese_reader_cfg = dict(input_columns=['problem','question_type'], output_column='final_answer')

eese_infer_cfg = dict(
    prompt_template=dict(
        type=PromptTemplate,
        template=dict(
            round=[
                dict(role='HUMAN', prompt=ANSWER_TEMPLATE),
            ]
        ),
    ),
    retriever=dict(type=ZeroRetriever),
    inferencer=dict(type=GenInferencer),
)


GRADER_TEMPLATE = """
As a grading expert, please score the candidates' answers based on the standard answers to the questions.

The following are some evaluation criteria:
1. Please refer to the standard answer given. You don't need to regenerate the answer to the question because the standard answer has already been given. You only need to determine whether the candidate's answer is consistent with the standard answer based on the form of the question. Don't try to answer the initial question. You can assume that the standard answer is definitely correct.
2. As the candidates' answers may differ in expression form from the standard answers, please understand the question and the standard answer before making a judgment, and then score the candidates' answers. However, be careful not to attempt to answer the original question.
3. Some answers may contain multiple items, such as multiple-choice questions, fill-in-the-blank questions, etc. As long as the answer is the same as the standard answer, it is sufficient. For multiple-choice questions and fill-in-the-blank questions, candidates must correctly answer all the corresponding options or blanks to be considered correct.
4. Some answers can be expressed in different ways. For instance, some answers might be mathematical expressions and some might be textual descriptions, as long as the meaning expressed is the same. Some formulas are expressed in different ways, but they are equivalent and correct.

If this question is a closed-ended one, please directly determine whether the candidate's answer is correct or not. If it is correct, please give 10 points; if it is incorrect, please give 0 points. Please give the score directly without any other explanations.
If this question is an open-ended one, please refer to the standard answers to score the candidates' answers. The scoring range is 0 to 10 points. Please directly give the final score without any explanation.

This is your task. Just answer the corresponding score. If there are mistakes, don't apologize or correct yourself. We just want to rate the answers.


< Original problem Begins >:\n{problem}\n< Original problem ends >\n\n
< Golden Goal Begins >:\n{final_answer}\n< Golden Goal Ends >\n\n
< question_type Begins >:\n{question_type}\n< question_type Ends >\n\n
< Prediction answer Begins >:\n{prediction}\n< Prediction End >\n\n

Determine the correctness of the examinee's answers.
""".strip()

# Evaluation configuration
eese_eval_cfg = dict(
    evaluator=dict(
        type=GenericLLMEvaluator,
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
        dataset_cfg=dict(
            type=EESEDataset,
            path='opencompass/eese',
            file_name = 'EESE.jsonl',
            reader_cfg=eese_reader_cfg,
        ),
        judge_cfg=dict(),
        dict_postprocessor=dict(type=eese_score_postprocess_dict),
    ),
    pred_role='BOT',
)


eese_datasets = [
    dict(
        type=EESEDataset,
        abbr='eese-llmjudge',
        path='opencompass/eese',
        file_name = 'EESE.jsonl',
        reader_cfg=eese_reader_cfg,
        infer_cfg=eese_infer_cfg,
        eval_cfg=eese_eval_cfg,
        mode='singlescore',
    )
]