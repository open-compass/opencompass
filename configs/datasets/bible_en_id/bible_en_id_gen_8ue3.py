from opencompass.openicl.icl_prompt_template import PromptTemplate
from opencompass.openicl.icl_retriever import BM25Retriever
from opencompass.openicl.icl_inferencer import GenInferencer
from opencompass.openicl.icl_evaluator import BleuEvaluator
from opencompass.datasets import BibleEnIdDataset


bibleEnId_reader_cfg = dict(
    input_columns=['en'],
    output_column='id',
    test_split='test')

bibleEnId_infer_cfg = dict(
    ice_template=dict(
        type=PromptTemplate,
        template=dict(
            begin='</E>',
            round=[
                dict(role='HUMAN', prompt='Translate the provided sentences from English to Indonesian while maintaining the original meaning and context: {en}?'),
                dict(role='BOT', prompt='{id}'),
            ]
        ),
        ice_token='</E>'),
    retriever=dict(type=BM25Retriever, ice_num=1),
    inferencer=dict(type=GenInferencer))


bibleEnId_eval_cfg = dict(
    evaluator=dict(type=BleuEvaluator),
    pred_role='BOT')

bibleEnId_datasets = [
    dict(
        type=BibleEnIdDataset,
        path='./data/bible_en_id',
        abbr='bible_en_id',
        reader_cfg=bibleEnId_reader_cfg,
        infer_cfg=bibleEnId_infer_cfg,
        eval_cfg=bibleEnId_eval_cfg)
]











