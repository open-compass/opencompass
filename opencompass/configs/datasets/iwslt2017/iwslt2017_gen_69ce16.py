from opencompass.openicl.icl_prompt_template import PromptTemplate
from opencompass.openicl.icl_retriever import BM25Retriever
from opencompass.openicl.icl_inferencer import GenInferencer
from opencompass.openicl.icl_evaluator import BleuEvaluator
from opencompass.datasets import IWSLT2017Dataset
from opencompass.utils.text_postprocessors import general_cn_postprocess

iwslt2017_reader_cfg = dict(
    input_columns='en', output_column='de', train_split='validation')

iwslt2017_infer_cfg = dict(
    ice_template=dict(type='PromptTemplate',
                    template='</E>{en} = {de}',
                    ice_token='</E>'),
    retriever=dict(type=BM25Retriever, ice_num=1),
    inferencer=dict(type=GenInferencer))

iwslt2017_eval_cfg = dict(
    evaluator=dict(type=BleuEvaluator),
    pred_role='BOT',
    pred_postprocessor=dict(type=general_cn_postprocess),
    dataset_postprocessor=dict(type=general_cn_postprocess))

iwslt2017_datasets = [
    dict(
        type=IWSLT2017Dataset,
        path='iwslt2017',
        name='iwslt2017-en-de',
        reader_cfg=iwslt2017_reader_cfg,
        infer_cfg=iwslt2017_infer_cfg,
        eval_cfg=iwslt2017_eval_cfg)
]
