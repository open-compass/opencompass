from opencompass.openicl.icl_inferencer import GenInferencer
from opencompass.openicl.icl_raw_prompt_template import RawPromptTemplate
from opencompass.openicl.icl_retriever import ZeroRetriever
from opencompass.datasets import MultiIFDataset, MultiIFEvaluator

multiif_reader_cfg = dict(
    input_columns=['dialogue'],
    output_column='reference',
)

multiif_infer_cfg = dict(
    prompt_template=dict(
        type=RawPromptTemplate,
        messages=[{'expand_column': 'dialogue'}],
        format_variables=False,
    ),
    retriever=dict(type=ZeroRetriever),
    inferencer=dict(
        type=GenInferencer,
        multiround=True,
    ),
)

multiif_eval_cfg = dict(
    evaluator=dict(type=MultiIFEvaluator),
    pred_role='BOT',
)

multiif_datasets = [
    dict(
        abbr='Multi-IF',
        type=MultiIFDataset,
        path='opencompass/MultiIF',
        reader_cfg=multiif_reader_cfg,
        infer_cfg=multiif_infer_cfg,
        eval_cfg=multiif_eval_cfg,
    ),
]
