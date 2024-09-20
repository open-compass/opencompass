from opencompass.openicl.icl_prompt_template import PromptTemplate
from opencompass.openicl.icl_retriever import MDLRetriever
from opencompass.openicl.icl_inferencer import PPLInferencer
from opencompass.openicl.icl_evaluator import AccEvaluator
from opencompass.datasets import commonsenseqaDataset

commonsenseqa_reader_cfg = dict(
    input_columns=['question', 'A', 'B', 'C', 'D', 'E'],
    output_column='answerKey',
    test_split='validation')

_ice_template = dict(
    type=PromptTemplate,
    template={
        'A': '</E>Answer the following question:\n{question}\nAnswer: {A}',
        'B': '</E>Answer the following question:\n{question}\nAnswer: {B}',
        'C': '</E>Answer the following question:\n{question}\nAnswer: {C}',
        'D': '</E>Answer the following question:\n{question}\nAnswer: {D}',
        'E': '</E>Answer the following question:\n{question}\nAnswer: {E}',
    },
    ice_token='</E>')

commonsenseqa_infer_cfg = dict(
    ice_template=_ice_template,
    retriever=dict(
        type=MDLRetriever,
        ice_num=8,
        candidate_num=30,
        select_time=10,
        seed=1,
        batch_size=12,
        ice_template=_ice_template),
    inferencer=dict(type=PPLInferencer))

commonsenseqa_eval_cfg = dict(evaluator=dict(type=AccEvaluator))

commonsenseqa_datasets = [
    dict(
        abbr='commonsense_qa',
        type=commonsenseqaDataset,
        path='opencompass/commonsense_qa',
        reader_cfg=commonsenseqa_reader_cfg,
        infer_cfg=commonsenseqa_infer_cfg,
        eval_cfg=commonsenseqa_eval_cfg)
]
