from opencompass.openicl.icl_prompt_template import PromptTemplate
from opencompass.openicl.icl_retriever import ZeroRetriever
from opencompass.openicl.icl_inferencer import GenInferencer
from opencompass.openicl.icl_evaluator import RougeEvaluator
from opencompass.datasets import XLSUMDataset, Xsum_postprocess

XLSum_reader_cfg = dict(input_columns=['text'], output_column='summary')

XLSum_infer_cfg = dict(
    prompt_template=dict(
        type=PromptTemplate,
        template='Document：{text}\n'
        'Based on the previous text, provide a brief single summary:'),
    retriever=dict(type=ZeroRetriever),
    inferencer=dict(type=GenInferencer))

XLSum_eval_cfg = dict(
    evaluator=dict(type=RougeEvaluator),
    pred_postprocessor=dict(type=Xsum_postprocess),
)

XLSum_datasets = [
    dict(
        type=XLSUMDataset,
        path='csebuetnlp/xlsum',
        reader_cfg=XLSum_reader_cfg,
        infer_cfg=XLSum_infer_cfg,
        eval_cfg=XLSum_eval_cfg)
]
