from opencompass.openicl.icl_prompt_template import PromptTemplate
from opencompass.openicl.icl_retriever import ZeroRetriever
from opencompass.openicl.icl_inferencer import GenInferencer
from opencompass.datasets import ClimaQADataset, generic_llmjudge_postprocess

from opencompass.evaluator import GenericLLMEvaluator

climaqa_silver_sets = [
    'mcq',
    'cloze',
    'ffq'
]

GRADER_TEMPLATE_mcq = """
    Please as a grading expert, judge whether the final answers given by the candidates below are consistent with the standard answers, that is, whether the candidates answered correctly.

    Here are some evaluation criteria:
    1. Please refer to the given standard answer. You don't need to re-generate the answer to the question because the standard answer has been given. You only need to judge whether the candidate's answer is consistent with the standard answer according to the form of the question. Don't try to answer the original question. You can assume that the standard answer is definitely correct.
    2. The answer may be one of the four options: a, b, c, or d. Only when the options given by prediction are strictly consistent with the answer, the prediction can be considered correct.
    3. If the prediction is given with 'The answer is:', please ignore the 'The answer is:', and only judge whether the candidate's answer is consistent with the standard answer.


    Please judge whether the following answers are consistent with the standard answer based on the above criteria. Grade the predicted answer of this new question as one of:
    A: CORRECT
    B: INCORRECT
    Just return the letters "A" or "B", with no text around it.

    Here is your task. Simply reply with either CORRECT, INCORRECT. Don't apologize or correct yourself if there was a mistake; we are just trying to grade the answer.


    <Original Question Begin>: \n{input}\n<Original Question End>\n\n
    <Gold Target Begin>: \n{target}\n<Gold Target End>\n\n
    <Predicted Answer Begin>: \n{prediction}\n<Predicted End>\n\n

    Judging the correctness of candidates' answers:
""".strip()

GRADER_TEMPLATE_cloze = """
    Please as a grading expert, judge whether the final answers given by the candidates below are consistent with the standard answers, that is, whether the candidates answered correctly.

    Here are some evaluation criteria:
    1. Please refer to the given standard answer. You don't need to re-generate the answer to the question because the standard answer has been given. You only need to judge whether the candidate's answer is consistent with the standard answer according to the form of the question. Don't try to answer the original question. You can assume that the standard answer is definitely correct.
    2. The form of the answer is a word or a phrase. Please strictly compare the prediction and the answer. Only when the prediction and the answer are exactly the same, will the prediction be considered correct; otherwise, it will be considered incorrect.
    3. If the prediction is given with 'The answer is:', please ignore the 'The answer is:' and only judge whether the candidate's answer is consistent with the standard answer.

    Please judge whether the following answers are consistent with the standard answer based on the above criteria. Grade the predicted answer of this new question as one of:
    A: CORRECT
    B: INCORRECT
    Just return the letters "A" or "B", with no text around it.

    Here is your task. Simply reply with either CORRECT, INCORRECT. Don't apologize or correct yourself if there was a mistake; we are just trying to grade the answer.


    <Original Question Begin>: \n{input}\n<Original Question End>\n\n
    <Gold Target Begin>: \n{target}\n<Gold Target End>\n\n
    <Predicted Answer Begin>: \n{prediction}\n<Predicted End>\n\n

    Judging the correctness of candidates' answers:
""".strip()

GRADER_TEMPLATE_ffq = """
    Please as a grading expert, judge whether the final answers given by the candidates below are consistent with the standard answers, that is, whether the candidates answered correctly.

    Here are some evaluation criteria:
    1. Please refer to the given standard answer. You don't need to re-generate the answer to the question because the standard answer has been given. You only need to judge whether the candidate's answer is consistent with the standard answer according to the form of the question. Don't try to answer the original question.
    2. Because the candidate's answer may be different from the standard answer in the form of expression, before making a judgment, please understand the question and the standard answer first, and then judge whether the candidate's answer is correct, but be careful not to try to answer the original question.
    3. The type of question is open-ended Q&A. Please compare whether the prediction is close enough to the meaning of the answer and whether the prediction covers each key point in the answer. If the prediction meets the above requirements, it can be considered very close to the answer.
    4. Some answers may be expressed in different ways, such as some answers may be a mathematical expression, some answers may be a textual description, as long as the meaning expressed is the same. And some formulas are expressed in different ways, but they are equivalent and correct.
    5. If the prediction is given with 'The answer is:', please ignore the 'The answer is:' and only judge whether the candidate's answer is very close to the standard answer.

    Please judge whether the following answers are close to the standard answer based on the above criteria. Grade the predicted answer of this new question as one of:
    A: very close to the answer
    B: not very close to the answer
    Just return the letters "A" or "B", with no text around it.

    Here is your task. Simply reply with either A or B. Don't apologize or correct yourself if there was a mistake; we are just trying to grade the answer.


    <Original Question Begin>: \n{input}\n<Original Question End>\n\n
    <Gold Target Begin>: \n{target}\n<Gold Target End>\n\n
    <Predicted Answer Begin>: \n{prediction}\n<Predicted End>\n\n

    Judging the correctness of candidates' answers:
""".strip()

climaqa_reader_cfg = dict(input_columns=['input'], output_column='target')

climaqa_datasets = []

for _task in climaqa_silver_sets:

    if _task == 'mcq':
        GRADER_TEMPLATE = GRADER_TEMPLATE_mcq
        infer_prompt = f"Think step by step, and when you provide the final answer, please use the prefix \"The answer is:\"without any modification. The question is multiple choice with a single correct answer, the final answer must only be the letter corresponding to the correct answer. For example, \"The answer is: a\"\n\nQ: {{input}}\nA: "
    if _task == 'ffq':
        GRADER_TEMPLATE = GRADER_TEMPLATE_ffq
        infer_prompt = f"Think step by step, and when you provide the final answer, please use the prefix \"The answer is:\".\n\nQ: {{input}}\nA: "
    if _task == 'cloze':
        GRADER_TEMPLATE = GRADER_TEMPLATE_cloze
        infer_prompt = f"Fill the <Mask> in the sentence. Think step by step, and when you provide the final answer, please use the prefix \"The answer is:\"without any modification, and provide the answer directly, with no formatting, no bolding, and no markup. For instance: \"The answer is: 42\" or \"The answer is: yes\".\n\nQ: {{input}}\nA: "

    climaqa_infer_cfg = dict(
        prompt_template=dict(
            type=PromptTemplate,
            template=dict(
                round=[
                    dict(
                        role='HUMAN',
                        prompt=infer_prompt,
                    )
                ]
            ),
        ),
        retriever=dict(type=ZeroRetriever),
        inferencer=dict(type=GenInferencer),
    )

    climaqa_eval_cfg = dict(
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
                type=ClimaQADataset,
                path='opencompass/ClimaQA-Silver',
                task=_task,
                abbr='ClimaQA_Silver_' + _task,
                reader_cfg=climaqa_reader_cfg,
            ),
            judge_cfg=dict(),
            dict_postprocessor=dict(type=generic_llmjudge_postprocess),
        ),
        pred_role='BOT',
    )

    climaqa_datasets.append(
        dict(
            abbr='ClimaQA_Silver_' + _task,
            type=ClimaQADataset,
            path='opencompass/ClimaQA-Silver',
            task=_task,
            reader_cfg=climaqa_reader_cfg,
            infer_cfg=climaqa_infer_cfg,
            eval_cfg=climaqa_eval_cfg,
        )
    )

