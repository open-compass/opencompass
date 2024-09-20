from opencompass.openicl.icl_prompt_template import PromptTemplate
from opencompass.openicl.icl_retriever import ZeroRetriever, FixKRetriever
from opencompass.openicl.icl_inferencer import GenInferencer
from opencompass.datasets import TriviaQADatasetV2, TriviaQAEvaluator


triviaqa_datasets = []
for k in [1]:
    triviaqa_reader_cfg = dict(
        input_columns=['question'], output_column='answer', train_split='train', test_split='validation')

    if k == 0:
        triviaqa_infer_cfg = dict(
            prompt_template=dict(
                type=PromptTemplate,
                template=dict(
                    round=[
                        dict(role='HUMAN', prompt='Q: {question}'),
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
                        dict(role='HUMAN', prompt='Q: {question}'),
                        dict(role='BOT', prompt='A: {answer}.\n'),
                    ]
                ),
            ),
            prompt_template=dict(
                type=PromptTemplate,
                template=dict(
                    begin='</E>',
                    round=[
                        dict(role='HUMAN', prompt='Q: {question}'),
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
        type=TriviaQADatasetV2,
        abbr=f'triviaqa_wiki_{k}shot',
        path='opencompass/trivia_qa',
        reader_cfg=triviaqa_reader_cfg,
        infer_cfg=triviaqa_infer_cfg,
        eval_cfg=triviaqa_eval_cfg)
    )
