from opencompass.openicl.icl_prompt_template import PromptTemplate
from opencompass.openicl.icl_retriever import ZeroRetriever
from opencompass.openicl.icl_inferencer import GenInferencer
from opencompass.openicl.icl_evaluator import AccEvaluator
from opencompass.datasets import CValuesDataset
from opencompass.utils.text_postprocessors import first_capital_postprocess

cvalues_reader_cfg = dict(
    input_columns=['prompt'],
    output_column='label',
    train_split='train',
    test_split='train',
)

cvalues_infer_cfg = dict(
    prompt_template=dict(
        type=PromptTemplate,
        template=dict(
            round=[dict(role='HUMAN', prompt='{prompt}请直接给出答案：\n')])),
    retriever=dict(type=ZeroRetriever),
    inferencer=dict(type=GenInferencer))

cvalues_eval_cfg = dict(
    evaluator=dict(type=AccEvaluator),
    pred_role='BOT',
    pred_postprocessor=dict(type=first_capital_postprocess),
)

cvalues_datasets = [
    dict(
        abbr='CValues-Responsibility',
        type=CValuesDataset,
        path='data/cvalues_responsibility_mc.jsonl',
        reader_cfg=cvalues_reader_cfg,
        infer_cfg=cvalues_infer_cfg,
        eval_cfg=cvalues_eval_cfg)
]
