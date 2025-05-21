"""
Summary: A config for KoR-Bench Evaluation.
Setting:
    Shot: 0-shot
    Evaluator:
        - CascadeEvaluator
            - korbenchEvaluator
            - GenericLLMEvaluator
    Repeat: 1
Avaliable Models:
    - Instruct/Chat Models
"""
from datasets import parallel
from opencompass.datasets.korbench.korbench import korbenchDataset, korbenchEvaluator
from opencompass.openicl.icl_inferencer import GenInferencer
from opencompass.openicl.icl_prompt_template import PromptTemplate
from opencompass.openicl.icl_retriever import ZeroRetriever
from opencompass.evaluator import GenericLLMEvaluator, CascadeEvaluator
from opencompass.datasets import generic_llmjudge_postprocess

categories = ['cipher', 'counterfactual', 'logic', 'operation', 'puzzle']

GRADER_TEMPLATE = """
    Please as a grading expert, judge whether the final answers given by the candidates below are consistent with the standard answers, that is, whether the candidates answered correctly. 

    Here are some evaluation criteria:
    1. Please refer to the given standard answer. You don't need to re-generate the answer to the question because the standard answer has been given. You only need to judge whether the candidate's answer is consistent with the standard answer according to the form of the question. Don't try to answer the original question. You can assume that the standard answer is definitely correct.
    2. Because the candidate's answer may be different from the standard answer in the form of expression, before making a judgment, please understand the question and the standard answer first, and then judge whether the candidate's answer is correct, but be careful not to try to answer the original question.
    3. Some answers may contain multiple items, such as multiple-choice questions, multiple-select questions, fill-in-the-blank questions, etc. As long as the answer is the same as the standard answer, it is enough. For multiple-select questions and multiple-blank fill-in-the-blank questions, the candidate needs to answer all the corresponding options or blanks correctly to be considered correct.
    4. Some answers may be expressed in different ways, such as some answers may be a mathematical expression, some answers may be a textual description, as long as the meaning expressed is the same. And some formulas are expressed in different ways, but they are equivalent and correct.
    5. If the prediction is given with \\boxed{}, please ignore the \\boxed{} and only judge whether the candidate's answer is consistent with the standard answer.

    Please judge whether the following answers are consistent with the standard answer based on the above criteria. Grade the predicted answer of this new question as one of:
    A: CORRECT 
    B: INCORRECT
    Just return the letters "A" or "B", with no text around it.

    Here is your task. Simply reply with either CORRECT, INCORRECT. Don't apologize or correct yourself if there was a mistake; we are just trying to grade the answer.


    <Original Question Begin>: \n{prompt}\n<Original Question End>\n\n
    <Gold Target Begin>: \n{answer}\n<Gold Target End>\n\n
    <Predicted Answer Begin>: \n{prediction}\n<Predicted End>\n\n

    Judging the correctness of candidates' answers:
""".strip()

korbench_0shot_single_datasets = []

for category in categories:
    # Prompt template
    prompt_template = dict(
        type=PromptTemplate,
        template=dict(
            begin=[
                dict(
                    role='HUMAN',
                    prompt=''
                )
            ],
            round=[
                dict(
                    role='HUMAN',
                    prompt='{prompt}'  # f-string
                )
            ]
        )
    )

    # Reader configuration
    reader_cfg = dict(
        input_columns=['prompt'],
        output_column='answer',
    )

    # Inference configuration
    infer_cfg = dict(
        prompt_template=prompt_template,
        retriever=dict(type=ZeroRetriever),
        inferencer=dict(type=GenInferencer),
    )

    # Evaluation configuration
    eval_cfg = dict(
        evaluator=dict(
            type=CascadeEvaluator,
            rule_evaluator=dict(
                type=korbenchEvaluator,
            ),
            llm_evaluator=dict(
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
                                prompt=GRADER_TEMPLATE
                            ),
                        ]),
                ),
                dataset_cfg=dict(
                    type=korbenchDataset,
                    path='opencompass/korbench',
                    prompt_mode='0_shot',
                    category=category,
                    reader_cfg=reader_cfg,
                ),
                judge_cfg=dict(),
                dict_postprocessor=dict(type=generic_llmjudge_postprocess),
            ),
            parallel=False,
        )
    )

    # Dataset
    korbench_dataset = dict(
        type=korbenchDataset,
        abbr=f'korbench_{category}',
        path='opencompass/korbench',
        prompt_mode='0_shot',
        category=category,
        reader_cfg=reader_cfg,
        infer_cfg=infer_cfg,
        eval_cfg=eval_cfg,
        n=1,
    )

    korbench_0shot_single_datasets.append(korbench_dataset)