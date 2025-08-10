from opencompass.openicl.icl_prompt_template import PromptTemplate
from opencompass.openicl.icl_retriever import ZeroRetriever
from opencompass.openicl.icl_inferencer import GenInferencer
from opencompass.openicl.icl_evaluator import BleuEvaluator
from opencompass.datasets import GovRepcrsDataset
from opencompass.utils.text_postprocessors import general_cn_postprocess

govrepcrs_reader_cfg = dict(
    input_columns='content',
    output_column='summary',
    train_split='test',
    test_split='test')

govrepcrs_infer_cfg = dict(
    prompt_template=dict(
        type=PromptTemplate,
        template=dict(
            begin=[
                dict(
                    role='SYSTEM',
                    fallback_role='HUMAN',
                    prompt=
                    'Please summarize the following English report in English:'
                ),
            ],
            round=[
                dict(role='HUMAN', prompt='{content}'),
                dict(role='BOT', prompt='{summary}'),
            ])),
    retriever=dict(type=ZeroRetriever),
    inferencer=dict(
        type=GenInferencer, batch_size=4, max_out_len=500, max_seq_len=8192))

govrepcrs_eval_cfg = dict(
    evaluator=dict(type=BleuEvaluator),
    pred_role='BOT',
    pred_postprocessor=dict(type=general_cn_postprocess),
    dataset_postprocessor=dict(type=general_cn_postprocess))

govrepcrs_datasets = [
    dict(
        type=GovRepcrsDataset,
        path='./data/govrep/',
        abbr='GovRepcrs',
        reader_cfg=govrepcrs_reader_cfg,
        infer_cfg=govrepcrs_infer_cfg,
        eval_cfg=govrepcrs_eval_cfg)
]
