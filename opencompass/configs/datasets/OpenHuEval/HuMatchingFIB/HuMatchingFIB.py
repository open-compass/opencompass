from mmengine.config import read_base

from opencompass.openicl.icl_prompt_template import PromptTemplate
from opencompass.openicl.icl_retriever import ZeroRetriever
from opencompass.openicl.icl_inferencer import GenInferencer
from opencompass.datasets.OpenHuEval.HuMatchingFIB import HuMatchingFIBDataset, HuMatchingFIBEvaluator

with read_base():
    from .HuMatchingFIB_setting import INSTRUCTION, DATA_PATH, DATA_VERSION

instruction = INSTRUCTION['prompt_template']
prompt_version = INSTRUCTION['version']

hu_matching_fib_reader_cfg = dict(
    input_columns=['question', 'options', 'hu_specific_dim'],
    output_column='reference')

hu_matching_fib_datasets = []

hu_matching_fib_infer_cfg = dict(
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

hu_matching_fib_eval_cfg = dict(evaluator=dict(type=HuMatchingFIBEvaluator))

hu_matching_fib_datasets.append(
    dict(
        abbr=f'hu_matching_fib_{DATA_VERSION}-prompt_{prompt_version}',
        type=HuMatchingFIBDataset,
        filepath=DATA_PATH,
        reader_cfg=hu_matching_fib_reader_cfg,
        infer_cfg=hu_matching_fib_infer_cfg,
        eval_cfg=hu_matching_fib_eval_cfg,
    ))
