from mmengine.config import read_base

from opencompass.openicl.icl_prompt_template import PromptTemplate
from opencompass.openicl.icl_retriever import ZeroRetriever
from opencompass.openicl.icl_inferencer import GenInferencer
from opencompass.datasets.OpenHuEval.HuProverbRea import HuProverbDataset2CQ, HuProverb_Evaluator_2CQ

with read_base():
    from .prompts import INSTRUCTIONS_DIRECT_QA

# currently we use English prompts with hu proverbs inserted
prompt_template_language = 'en'
dataset_path = '/mnt/hwfile/opendatalab/gaojunyuan/shared_data/OpenHuEval/data/HuProverbRea/HuProverbRea_250127'

HuProverbRea_reader_cfg = dict(input_columns=['hu_text', 'context', 'en_expl', 'hu_expl', 'option1', 'option2'],
                         output_column='out')

HuProverbRea_datasets = []
instruction = INSTRUCTIONS_DIRECT_QA[prompt_template_language]
HuProverbRea_infer_cfg = dict(
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

HuProverbRea_eval_cfg = dict(evaluator=dict(type=HuProverb_Evaluator_2CQ))

HuProverbRea_datasets.append(
    dict(
        abbr=f'HuProverbRea_2CQ_{prompt_template_language}',
        type=HuProverbDataset2CQ,
        path=dataset_path,
        reader_cfg=HuProverbRea_reader_cfg,
        infer_cfg=HuProverbRea_infer_cfg,
        eval_cfg=HuProverbRea_eval_cfg,
    )
)
