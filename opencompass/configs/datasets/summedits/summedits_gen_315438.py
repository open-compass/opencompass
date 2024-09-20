from opencompass.openicl.icl_prompt_template import PromptTemplate
from opencompass.openicl.icl_retriever import ZeroRetriever
from opencompass.openicl.icl_inferencer import GenInferencer
from opencompass.openicl.icl_evaluator import AccEvaluator
from opencompass.datasets import SummeditsDataset_V2
from opencompass.utils.text_postprocessors import first_capital_postprocess

summedits_reader_cfg = dict(
    input_columns=['doc', 'summary'], output_column='label')

summedits_infer_cfg = dict(
    prompt_template=dict(
        type=PromptTemplate,
        template=dict(round=[
            dict(
                role='HUMAN',
                prompt=
                """Given the document below, you have to determine if "Yes" or "No", the summary is factually consistent with the document.

Document:
{doc}

Summary:
{summary}

Question:
Is the summary factually consistent with the document?
A. Yes
B. No

Answer:"""
            ),
        ])),
    retriever=dict(type=ZeroRetriever),
    inferencer=dict(type=GenInferencer))

summedits_eval_cfg = dict(
    evaluator=dict(type=AccEvaluator),
    pred_role='BOT',
    pred_postprocessor=dict(type=first_capital_postprocess),
)

summedits_datasets = [
    dict(
        abbr='summedits',
        type=SummeditsDataset_V2,
        path='opencompass/summedits',
        reader_cfg=summedits_reader_cfg,
        infer_cfg=summedits_infer_cfg,
        eval_cfg=summedits_eval_cfg)
]
