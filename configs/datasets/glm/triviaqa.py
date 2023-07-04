from opencompass.openicl.icl_prompt_template import PromptTemplate
from opencompass.openicl.icl_retriever import ZeroRetriever
from opencompass.openicl.icl_inferencer import GenInferencer
from opencompass.datasets import TriviaQADataset, TriviaQAEvaluator

triviaqa_reader_cfg = dict(
    input_columns=['question'],
    output_column='answer',
    train_split='dev',
    test_split='dev')

triviaqa_infer_cfg = dict(
    ice_template=dict(
        type=PromptTemplate,
        template='Q: </Q>\nA: </A>',
        column_token_map={
            'question': '</Q>',
            'answer': '</A>'
        }),
    prompt_template=dict(
        type=PromptTemplate,
        template='</E>Question: </Q> Answer:',
        column_token_map={
            'question': '</Q>',
            'answer': '</A>'
        },
        ice_token='</E>'),
    retriever=dict(type=ZeroRetriever),
    inferencer=dict(type=GenInferencer, max_out_len=50))

triviaqa_eval_cfg = dict(evaluator=dict(type=TriviaQAEvaluator))

triviaqa_datasets = [
    dict(
        type=TriviaQADataset,
        abbr='triviaqa',
        path='./data/triviaqa/',
        reader_cfg=triviaqa_reader_cfg,
        infer_cfg=triviaqa_infer_cfg,
        eval_cfg=triviaqa_eval_cfg)
]
