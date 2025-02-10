from mmengine.config import read_base

from opencompass.openicl.icl_prompt_template import PromptTemplate
from opencompass.openicl.icl_retriever import ZeroRetriever
from opencompass.openicl.icl_inferencer import GenInferencer
from opencompass.datasets.OpenHuEval.HuMatchingFIB import HuMatchingFIBDataset, HuMatchingFIBEvaluator

with read_base():
    from .HuMatchingFIB_setting import INSTRUCTIONS, DATA_PATH, DATA_VERSION

PROMPT_LANGUAGES = [
    'en',
    # 'hu',
]

HuMatchingFIB_reader_cfg = dict(
    input_columns=['question', 'options', 'hu_specific_dim'],
    output_column='reference')

HuMatchingFIB_datasets = []

for lang in PROMPT_LANGUAGES:
    instruction = INSTRUCTIONS[lang]

    HuMatchingFIB_infer_cfg = dict(
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

    HuMatchingFIB_eval_cfg = dict(evaluator=dict(type=HuMatchingFIBEvaluator))

    HuMatchingFIB_datasets.append(
        dict(
            abbr=f'OpenHuEval_HuMatchingFIB_{DATA_VERSION}-prompt_{lang}',
            type=HuMatchingFIBDataset,
            filepath=DATA_PATH,
            reader_cfg=HuMatchingFIB_reader_cfg,
            infer_cfg=HuMatchingFIB_infer_cfg,
            eval_cfg=HuMatchingFIB_eval_cfg,
        ))
