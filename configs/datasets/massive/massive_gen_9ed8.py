from opencompass.openicl.icl_prompt_template import PromptTemplate
from opencompass.openicl.icl_retriever import BM25Retriever
from opencompass.openicl.icl_inferencer import GenInferencer
from opencompass.openicl.icl_evaluator import MassiveEvaluator
from opencompass.datasets import MassiveDataset

massive_reader_cfg = dict(
    input_columns=['text'],
    output_column='output',
    test_split='test')

massive_infer_cfg = dict(
    ice_template=dict(
        type=PromptTemplate,
        template=dict(
            begin='</E>',
            round=[
                dict(role='HUMAN', prompt='Read the sentence and identify each entity, output the entity type and name with template "entity_type_1 : entity_name_1 ; entity_type_2 : entity_name_2": {text}?'),
                dict(role='BOT', prompt='{output}'),
            ]
        ),
        ice_token='</E>'),
    retriever=dict(type=BM25Retriever, ice_num=1),
    inferencer=dict(type=GenInferencer, max_out_len=1024))


massive_eval_cfg = dict(
    evaluator=dict(type=MassiveEvaluator),
    pred_role="BOT")


massive_datasets = [
    dict(
        type=MassiveDataset,
        abbr='massive',
        path='./data/massive/',
        reader_cfg=massive_reader_cfg,
        infer_cfg=massive_infer_cfg,
        eval_cfg=massive_eval_cfg)
]
