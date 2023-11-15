from opencompass.openicl.icl_prompt_template import PromptTemplate
from opencompass.openicl.icl_retriever import BM25Retriever
from opencompass.openicl.icl_inferencer import GenInferencer
from opencompass.openicl.icl_evaluator import BleuEvaluator
from opencompass.datasets import WikilinguaDocDataset


wikilinguadoc_reader_cfg = dict(
    input_columns=['en'],
    output_column='th',
    test_split='test')

wikilinguadoc_infer_cfg = dict(
    ice_template=dict(
        type=PromptTemplate,
        template=dict(
            begin='</E>',
            round=[
                dict(role='HUMAN', prompt='Translate the provided sentences from English to Thai while maintaining the original meaning and context: {en}?'),
                dict(role='BOT', prompt='{th}'),
            ]
        ),
        ice_token='</E>'),
    retriever=dict(type=BM25Retriever, ice_num=1),
    inferencer=dict(type=GenInferencer))


wikilinguadoc_eval_cfg = dict(
    evaluator=dict(type=BleuEvaluator),
    pred_role='BOT')

wikilinguadoc_datasets = [
    dict(
        type=WikilinguaDocDataset,
        path='./data/wikilingua_doc',
        abbr='wikilinguadoc',
        reader_cfg=wikilinguadoc_reader_cfg,
        infer_cfg=wikilinguadoc_infer_cfg,
        eval_cfg=wikilinguadoc_eval_cfg)
]
