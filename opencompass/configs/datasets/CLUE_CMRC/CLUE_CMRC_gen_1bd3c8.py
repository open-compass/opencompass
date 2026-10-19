from opencompass.openicl.icl_prompt_template import PromptTemplate
from opencompass.openicl.icl_retriever import ZeroRetriever
from opencompass.openicl.icl_inferencer import GenInferencer
from opencompass.openicl.icl_evaluator import EMEvaluator
from opencompass.datasets import CMRCDataset, cmrc_postprocess

CMRC_reader_cfg = dict(
    input_columns=['question', 'context'], output_column='answers')

CMRC_infer_cfg = dict(
    prompt_template=dict(
        type=PromptTemplate,
        template=dict(round=[
            dict(
                role='HUMAN',
                prompt='根据文章回答问题。你的答案应该尽可能简练，请以 ‘答案是’ 开头的句式作答。\n文章：{context}\n问：{question}\n答：'),
        ])),
    retriever=dict(type=ZeroRetriever),
    inferencer=dict(type=GenInferencer))

CMRC_eval_cfg = dict(
    evaluator=dict(type=EMEvaluator),
    pred_role='BOT',
    pred_postprocessor=dict(type=cmrc_postprocess),
)

CMRC_datasets = [
    dict(
        type=CMRCDataset,
        abbr='CMRC_dev',
        path='opencompass/cmrc_dev',
        reader_cfg=CMRC_reader_cfg,
        infer_cfg=CMRC_infer_cfg,
        eval_cfg=CMRC_eval_cfg),
]
