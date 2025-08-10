from opencompass.openicl.icl_prompt_template import PromptTemplate
from opencompass.openicl.icl_retriever import ZeroRetriever
from opencompass.openicl.icl_inferencer import ChatInferencer
from opencompass.datasets import SciCodeDataset, SciCodeEvaluator


SciCode_reader_cfg = dict(input_columns=['prompt'], output_column=None)

SciCode_infer_cfg = dict(
    ice_template=dict(
        type=PromptTemplate,
        template='',
        ),

    retriever=dict(type=ZeroRetriever),
    inferencer=dict(type=ChatInferencer, infer_mode='every', max_out_len=4096))

SciCode_eval_cfg = dict(evaluator=dict(type=SciCodeEvaluator, dataset_path='./data/scicode', with_bg=False))

SciCode_datasets = [
    dict(
        abbr='SciCode',
        type=SciCodeDataset,
        path='./data/scicode',
        with_bg=False,
        reader_cfg=SciCode_reader_cfg,
        infer_cfg=SciCode_infer_cfg,
        eval_cfg=SciCode_eval_cfg)
]
