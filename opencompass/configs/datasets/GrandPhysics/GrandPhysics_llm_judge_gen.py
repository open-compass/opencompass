from opencompass.openicl.icl_prompt_template import PromptTemplate
from opencompass.openicl.icl_retriever import ZeroRetriever
from opencompass.openicl.icl_inferencer import GenInferencer
from opencompass.datasets import (
    GrandPhysicsDataset,
    generic_llmjudge_postprocess,
)
from opencompass.evaluator import GenericLLMEvaluator

GRADER_TEMPLATE = """
    Please as a grading expert, judge whether the final answers given by the candidates below are consistent with the standard answers, that is, whether the candidates answered correctly.

    Here are some evaluation criteria:
    1. Please refer to the given standard answer. You don't need to re-generate the answer to the question because the standard answer has been given. You only need to judge whether the candidate's answer is consistent with the standard answer according to the form of the question. Don't try to answer the original question. You can assume that the standard answer is definitely correct.
    2. Because the candidate's answer may be different from the standard answer in the form of expression, before making a judgment, please understand the question and the standard answer first, and then judge whether the candidate's answer is correct, but be careful not to try to answer the original question.
    3. Some questions may include multiple sub questions and sub answers. Each sub answer is given after a guide character in the form of <Answer 1:> or <Answer 2:>, etc. Please note that only when all sub predictions given in prediction correspond one-to-one with the answer and are all correct, will the prediction be considered correct; otherwise, it will be considered incorrect.
    4. Some answers may be expressed in different ways, such as some answers may be a mathematical expression, some answers may be a textual description, as long as the meaning expressed is the same. And some formulas are expressed in different ways, but they are equivalent and correct.
    5. The final answers in the prediction are generally given with \\boxed{}. If you cannot find sufficient \\boxed{} in the prediction, please try to find matching answers from other places within the prediction as much as possible.

    Please judge whether the following answers are consistent with the standard answer based on the above criteria. Grade the predicted answer of this new question as one of:
    A: All Sub Predictions Are Correct
    B: Not Every Sub Predictions is Correct
    Just return the letters "A" or "B", with no text around it.

    Here is your task. Simply reply with either A, B. Don't apologize or correct yourself if there was a mistake; we are just trying to grade the answer.

    <Original Question Begin>: \n{input}\n<Original Question End>\n\n
    <Gold Target Begin>: \n{target}\n<Gold Target End>\n\n
    <Predicted Answer Begin>: \n{prediction}\n<Predicted End>\n\n

    Judging the correctness of candidates' answers:
""".strip()

grand_physics_reader_cfg = dict(input_columns=['input'], output_column='target')


grand_physics_infer_cfg = dict(
    prompt_template=dict(
        type=PromptTemplate,
        template=dict(
            round=[
                dict(
                    role='HUMAN',
                    prompt=f'Answer the given question step by step. Begin by explaining your reasoning process clearly. Conclude by providing the final answers at the end in LaTeX boxed format. Think step by step before answering. It should be noted that the question may include multiple sub questions, please ensure that each question is answered in order.\n\nQ: {{input}}\nA: ',
                )
            ]
        ),
    ),
    retriever=dict(type=ZeroRetriever),
    inferencer=dict(type=GenInferencer),
)

grand_physics_eval_cfg = dict(
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
            type=GrandPhysicsDataset,
            path='opencompass/GrandPhysics',
            abbr='GrandPhysics',
            reader_cfg=grand_physics_reader_cfg,
        ),
        judge_cfg=dict(),
        dict_postprocessor=dict(type=generic_llmjudge_postprocess),
    ),
    pred_role='BOT',
)

grand_physics_dataset = [
    dict(
        abbr='GrandPhysics',
        type=GrandPhysicsDataset,
        path='opencompass/GrandPhysics',
        reader_cfg=grand_physics_reader_cfg,
        infer_cfg=grand_physics_infer_cfg,
        eval_cfg=grand_physics_eval_cfg,
    )
]

