from opencompass.openicl.icl_prompt_template import PromptTemplate
from opencompass.openicl.icl_retriever import ZeroRetriever
from opencompass.openicl.icl_inferencer import GenInferencer
from opencompass.openicl.icl_evaluator import EMEvaluator
from opencompass.datasets import DRCDDataset

DRCD_reader_cfg = dict(
    input_columns=['question', 'context'], output_column='answers')

DRCD_infer_cfg = dict(
    prompt_template=dict(
        type=PromptTemplate,
        template=dict(round=[
            dict(
                role='HUMAN',
                prompt='文章：{context}\n根据上文，回答如下问题：\n{question}\n答：'),
        ])),
    retriever=dict(type=ZeroRetriever),
    inferencer=dict(type=GenInferencer))

DRCD_eval_cfg = dict(
    evaluator=dict(type=EMEvaluator),
    pred_role='BOT',
)

DRCD_datasets = [
    dict(
        type=DRCDDataset,
        abbr='DRCD_dev',
        path='opencompass/drcd_dev',
        reader_cfg=DRCD_reader_cfg,
        infer_cfg=DRCD_infer_cfg,
        eval_cfg=DRCD_eval_cfg),
]
