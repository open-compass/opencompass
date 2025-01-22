from mmengine.config import read_base

from opencompass.openicl.icl_prompt_template import PromptTemplate
from opencompass.openicl.icl_retriever import ZeroRetriever
from opencompass.openicl.icl_inferencer import GenInferencer
from opencompass.datasets.OpenHuEval.HuStandardFIB import HuStandardFIBDataset, HuStandardFIBEvaluator 

with read_base():
    from .HuStandardFIB_setting import INSTRUCTIONS, DATASET_PATH

ALL_LANGUAGES = ['hu']
PROMPT_VERSION = INSTRUCTIONS['version']

FIB2_reader_cfg = dict(input_columns=['question', 'subject'],
                         output_column='reference')

FIB2_datasets = []
for lan in ALL_LANGUAGES:
    instruction = INSTRUCTIONS[lan]
    FIB2_infer_cfg = dict(
        prompt_template=dict(
            type=PromptTemplate,
            template=dict(
                begin='</E>',
                round=[
                    dict(
                        role='HUMAN',
                        prompt=instruction
                    ),
                ],
            ),
            ice_token='</E>',
        ),
        retriever=dict(type=ZeroRetriever),
        inferencer=dict(type=GenInferencer),
    )

    FIB2_eval_cfg = dict(evaluator=dict(type=HuStandardFIBEvaluator))

    FIB2_datasets.append(
        dict(
            abbr=f'HuStandardFIB-{lan}-1shot-{PROMPT_VERSION}',
            type=HuStandardFIBDataset,
            path=DATASET_PATH,
            lan=lan,
            reader_cfg=FIB2_reader_cfg,
            infer_cfg=FIB2_infer_cfg,
            eval_cfg=FIB2_eval_cfg,
        )
    )
