from opencompass.openicl.icl_prompt_template import PromptTemplate
from opencompass.openicl.icl_retriever import ZeroRetriever, FixKRetriever
from opencompass.openicl.icl_inferencer import GenInferencer
from opencompass.datasets import TriviaQADataset, TriviaQAEvaluator

triviaqa_datasets = []
for k in [0, 1, 5]:
    triviaqa_reader_cfg = dict(
        input_columns=['question'], output_column='answer', train_split='test', test_split='dev')

    if k == 0:
        triviaqa_infer_cfg = dict(
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
        triviaqa_infer_cfg = dict(
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

    triviaqa_eval_cfg = dict(evaluator=dict(type=TriviaQAEvaluator), pred_role='BOT')

    triviaqa_datasets.append(
        dict(
            type=TriviaQADataset,
            abbr='triviaqa' if k == 0 else f'triviaqa_{k}shot',
            path='opencompass/trivia_qa',
            reader_cfg=triviaqa_reader_cfg,
            infer_cfg=triviaqa_infer_cfg,
            eval_cfg=triviaqa_eval_cfg)
    )
