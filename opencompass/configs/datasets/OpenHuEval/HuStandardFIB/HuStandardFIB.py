from mmengine.config import read_base

from opencompass.openicl.icl_prompt_template import PromptTemplate
from opencompass.openicl.icl_retriever import ZeroRetriever
from opencompass.openicl.icl_inferencer import GenInferencer
from opencompass.datasets.OpenHuEval.HuStandardFIB import HuStandardFIBDataset, HuStandardFIBEvaluator

with read_base():
    from .HuStandardFIB_setting import INSTRUCTION, DATA_PATH, DATA_VERSION

instruction = INSTRUCTION['prompt_template']
prompt_version = INSTRUCTION['version']

hu_standard_fib_reader_cfg = dict(input_columns=['question', 'subject'],
                                  output_column='reference')

hu_standard_fib_datasets = []

hu_standard_fib_infer_cfg = dict(
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

hu_standard_fib_eval_cfg = dict(evaluator=dict(type=HuStandardFIBEvaluator))

hu_standard_fib_datasets.append(
    dict(
        abbr=f'hu_standard_fib_{DATA_VERSION}-prompt_{prompt_version}',
        type=HuStandardFIBDataset,
        filepath=DATA_PATH,
        reader_cfg=hu_standard_fib_reader_cfg,
        infer_cfg=hu_standard_fib_infer_cfg,
        eval_cfg=hu_standard_fib_eval_cfg,
    ))
