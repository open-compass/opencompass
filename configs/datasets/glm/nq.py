from opencompass.openicl.icl_prompt_template import PromptTemplate
from opencompass.openicl.icl_retriever import ZeroRetriever
from opencompass.openicl.icl_inferencer import GenInferencer
from opencompass.datasets import NaturalQuestionDataset, NQEvaluator

nq_reader_cfg = dict(
    input_columns=['question'], output_column='answer', train_split='test')

nq_infer_cfg = dict(
    ice_template=dict(
        type=PromptTemplate,
        template="Q: </Q>?\nA: </A>",
        column_token_map={
            'question': '</Q>',
            'answer': '</A>'
        }),
    prompt_template=dict(
        type=PromptTemplate,
        template="</E>Question: </Q>? Answer: ",
        column_token_map={
            'question': '</Q>',
            'answer': '</A>'
        },
        ice_token='</E>'),
    retriever=dict(type=ZeroRetriever),
    inferencer=dict(type=GenInferencer))

nq_eval_cfg = dict(evaluator=dict(type=NQEvaluator))

nq_datasets = [
    dict(
        type=NaturalQuestionDataset,
        abbr='nq',
        path='/mnt/petrelfs/wuzhiyong/datasets/nq/',
        reader_cfg=nq_reader_cfg,
        infer_cfg=nq_infer_cfg,
        eval_cfg=nq_eval_cfg)
]
