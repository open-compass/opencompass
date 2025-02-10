from mmengine.config import read_base

from opencompass.openicl.icl_prompt_template import PromptTemplate
from opencompass.openicl.icl_retriever import ZeroRetriever
from opencompass.openicl.icl_inferencer import GenInferencer
from opencompass.datasets.OpenHuEval.HuStandardFIB import HuStandardFIBDataset, HuStandardFIBEvaluator

with read_base():
    from .HuStandardFIB_setting import INSTRUCTIONS, DATA_PATH, DATA_VERSION

prompt_lang = 'en'
instruction = INSTRUCTIONS[prompt_lang]

HuStandardFIB_reader_cfg = dict(input_columns=['question', 'subject'],
                                output_column='reference')

HuStandardFIB_datasets = []

HuStandardFIB_infer_cfg = dict(
    prompt_template=dict(
        type=PromptTemplate,
        template=dict(
            begin='</E>',
            round=[
                dict(role='HUMAN', prompt=instruction),
            ],
        ),
        ice_token='</E>',
    ),
    retriever=dict(type=ZeroRetriever),
    inferencer=dict(type=GenInferencer),
)

HuStandardFIB_eval_cfg = dict(evaluator=dict(type=HuStandardFIBEvaluator))

HuStandardFIB_datasets.append(
    dict(
        abbr=f'OpenHuEval_HuStandardFIB_{DATA_VERSION}-prompt_{prompt_lang}',
        type=HuStandardFIBDataset,
        filepath=DATA_PATH,
        reader_cfg=HuStandardFIB_reader_cfg,
        infer_cfg=HuStandardFIB_infer_cfg,
        eval_cfg=HuStandardFIB_eval_cfg,
    ))
