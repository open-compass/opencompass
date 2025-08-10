from opencompass.openicl.icl_prompt_template import PromptTemplate
from opencompass.openicl.icl_retriever import ZeroRetriever, FixKRetriever
from opencompass.openicl.icl_inferencer import GenInferencer
from opencompass.datasets import NaturalQuestionDataset, NQEvaluator

nq_datasets = []
for k in [0, 1, 5]:
    nq_reader_cfg = dict(
        input_columns=['question'], output_column='answer', train_split='dev')

    if k == 0:
        nq_infer_cfg = dict(
            prompt_template=dict(
                type=PromptTemplate,
                template=dict(
                    round=[
                        dict(role='HUMAN', prompt='Answer these questions, your answer should be as simple as possible, start your answer with the prompt \'The answer is \'.\nQ: {question}?'),
                        dict(role='BOT', prompt='A:'),
                    ]
                )
            ),
            retriever=dict(type=ZeroRetriever),
            inferencer=dict(type=GenInferencer, max_out_len=50)
        )
    else:
        nq_infer_cfg = dict(
            ice_template=dict(
                type=PromptTemplate,
                template=dict(
                    round=[
                        dict(role='HUMAN', prompt='Answer the question, your answer should be as simple as possible, start your answer with the prompt \'The answer is \'.\nQ: {question}?'),
                        dict(role='BOT', prompt='A: The answer is {answer}.\n'),
                    ]
                ),
            ),
            prompt_template=dict(
                type=PromptTemplate,
                template=dict(
                    begin='</E>',
                    round=[
                        dict(role='HUMAN', prompt='Answer the question, your answer should be as simple as possible, start your answer with the prompt \'The answer is \'.\nQ: {question}?'),
                        dict(role='BOT', prompt='A:'),
                    ]
                ),
                ice_token='</E>',
            ),
            retriever=dict(type=FixKRetriever, fix_id_list=list(range(k))),
            inferencer=dict(type=GenInferencer, max_out_len=50),
        )

    nq_eval_cfg = dict(evaluator=dict(type=NQEvaluator), pred_role='BOT')

    nq_datasets.append(
        dict(
            type=NaturalQuestionDataset,
            abbr='nq' if k == 0 else f'nq_{k}shot',
            path='opencompass/natural_question',
            reader_cfg=nq_reader_cfg,
            infer_cfg=nq_infer_cfg,
            eval_cfg=nq_eval_cfg)
    )
