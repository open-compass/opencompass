from mmengine.config import read_base

from opencompass.openicl.icl_prompt_template import PromptTemplate
from opencompass.openicl.icl_retriever import ZeroRetriever
from opencompass.openicl.icl_inferencer import GenInferencer
from opencompass.datasets.OpenHuEval.HuMatchingFIB import HuMatchingFIBDataset, HuMatchingFIBEvaluator

with read_base():
    from .HuMatchingFIB_setting import INSTRUCTIONS, DATASET_PATH

ALL_LANGUAGES = ['hu']
PROMPT_VERSION = INSTRUCTIONS['version']

FIB1_reader_cfg = dict(input_columns=['question', 'subject'],
                         output_column='reference')

FIB1_datasets = []
for lan in ALL_LANGUAGES:
    instruction = INSTRUCTIONS[lan]
    FIB1_infer_cfg = dict(
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

    FIB1_eval_cfg = dict(evaluator=dict(type=HuMatchingFIBEvaluator))

    FIB1_datasets.append(
        dict(
            abbr=f'nkp_FIB1_humanities-{lan}-1shot-{PROMPT_VERSION}',
            type=HuMatchingFIBDataset,
            path=DATASET_PATH,
            lan=lan,
            reader_cfg=FIB1_reader_cfg,
            infer_cfg=FIB1_infer_cfg,
            eval_cfg=FIB1_eval_cfg,
        )
    )
