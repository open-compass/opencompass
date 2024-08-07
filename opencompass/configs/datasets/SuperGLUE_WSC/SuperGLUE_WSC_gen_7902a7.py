from opencompass.openicl.icl_prompt_template import PromptTemplate
from opencompass.openicl.icl_retriever import ZeroRetriever
from opencompass.openicl.icl_inferencer import GenInferencer
from opencompass.openicl.icl_evaluator import AccEvaluator
from opencompass.datasets import WSCDatasetV2
from opencompass.utils.text_postprocessors import first_capital_postprocess

WSC_reader_cfg = dict(
    input_columns=['span1', 'span2', 'text'],
    output_column='label',
)

WSC_infer_cfg = dict(
    prompt_template=dict(
        type=PromptTemplate,
        template=dict(round=[
            dict(
                role='HUMAN',
                prompt=
                "{text}\nIs '{span1}' and '{span2}' refers to the same entity in the above sentence?\nA. Yes\nB. No\nAnswer:"
            ),
        ]),
    ),
    retriever=dict(type=ZeroRetriever),
    inferencer=dict(type=GenInferencer),
)

WSC_eval_cfg = dict(
    evaluator=dict(type=AccEvaluator),
    pred_role='BOT',
    pred_postprocessor=dict(type=first_capital_postprocess),
)

WSC_datasets = [
    dict(
        abbr='WSC',
        type=WSCDatasetV2,
        path='./data/SuperGLUE/WSC/val.jsonl',
        reader_cfg=WSC_reader_cfg,
        infer_cfg=WSC_infer_cfg,
        eval_cfg=WSC_eval_cfg,
    )
]
