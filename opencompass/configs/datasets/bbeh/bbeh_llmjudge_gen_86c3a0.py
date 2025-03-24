import os
from opencompass.openicl.icl_prompt_template import PromptTemplate
from opencompass.openicl.icl_retriever import ZeroRetriever
from opencompass.openicl.icl_inferencer import GenInferencer
from opencompass.datasets import (
    BBEHDataset,
    generic_llmjudge_postprocess,
)
from opencompass.evaluator import GenericLLMEvaluator

bbeh_reader_cfg = dict(input_columns=['input'], output_column='target')


bbeh_multiple_choice_sets = [
    'bbeh_boolean_expressions',
    'bbeh_disambiguation_qa',
    'bbeh_geometric_shapes',
    'bbeh_hyperbaton',
    'bbeh_movie_recommendation',
    'bbeh_nycc',
    'bbeh_shuffled_objects',
]

bbeh_free_form_sets = [
    'bbeh_boardgame_qa',
    'bbeh_buggy_tables',
    'bbeh_causal_understanding',
    'bbeh_dyck_languages',
    'bbeh_linguini',
    'bbeh_multistep_arithmetic',
    'bbeh_object_counting',
    'bbeh_object_properties',
    'bbeh_sarc_triples',
    'bbeh_spatial_reasoning',
    'bbeh_sportqa',
    'bbeh_temporal_sequence',
    'bbeh_time_arithmetic',
    'bbeh_web_of_lies',
    'bbeh_word_sorting',
    'bbeh_zebra_puzzles',
]

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


    <Original Question Begin>: \n{input}\n<Original Question End>\n\n
    <Gold Target Begin>: \n{target}\n<Gold Target End>\n\n
    <Predicted Answer Begin>: \n{prediction}\n<Predicted End>\n\n

    Judging the correctness of candidates' answers:
""".strip()

bbeh_datasets = []
for _name in bbeh_multiple_choice_sets + bbeh_free_form_sets:
    bbeh_infer_cfg = dict(
        prompt_template=dict(
            type=PromptTemplate,
            template=dict(
                round=[
                    dict(
                        role='HUMAN',
                        prompt=f"Think step by step, and when you provide the final answer, please use the prefix \"The answer is:\"without any modification, and provide the answer directly, with no formatting, no bolding, and no markup. For instance: \"The answer is: 42\" or \"The answer is: yes\". If the question is multiple choice with a single correct answer, the final answer must only be the letter corresponding to the correct answer. For example, \"The answer is: (a)\"\n\nQ: {{input}}\nA: ",
                    )
                ]
            ),
        ),
        retriever=dict(type=ZeroRetriever),
        inferencer=dict(type=GenInferencer),
    )
    bbeh_eval_cfg = dict(
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
                type=BBEHDataset,
                path='opencompass/bbeh',
                name=_name,
                abbr=_name,
                reader_cfg=bbeh_reader_cfg,
            ),
            judge_cfg=dict(),
            dict_postprocessor=dict(type=generic_llmjudge_postprocess),
        ),
        pred_role='BOT',
    )

    bbeh_datasets.append(
        dict(
            type=BBEHDataset,
            path='opencompass/bbeh',
            name=_name,
            abbr=_name,
            reader_cfg=bbeh_reader_cfg,
            infer_cfg=bbeh_infer_cfg,
            eval_cfg=bbeh_eval_cfg,
        )
    )