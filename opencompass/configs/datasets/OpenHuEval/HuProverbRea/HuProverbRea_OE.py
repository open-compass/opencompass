from mmengine.config import read_base

from opencompass.openicl.icl_prompt_template import PromptTemplate
from opencompass.openicl.icl_retriever import ZeroRetriever
from opencompass.openicl.icl_inferencer import GenInferencer
from opencompass.datasets.OpenHuEval.HuProverbRea import HuProverbDatasetOE, HuProverb_Evaluator_OE

with read_base():
    from .HuProverbRea_setting import INSTRUCTIONS_OE_DIR_QA, DATA_PATH, DATA_VERSION, judge_prompt_template

# currently we use English prompts with hu proverbs inserted
prompt_template_language = 'en'

HuProverbRea_reader_cfg = dict(
    input_columns=[
        'hu_text', 'context', 'en_expl', 'hu_expl', 'option1', 'option2'
    ],
    output_column='out',
)

HuProverbRea_datasets = []
instruction = INSTRUCTIONS_OE_DIR_QA[prompt_template_language]
HuProverbRea_infer_cfg = dict(
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

HuProverbRea_eval_cfg = dict(evaluator=dict(
    type=HuProverb_Evaluator_OE,
    judge_prompt_template=judge_prompt_template,
))

HuProverbRea_datasets.append(
    dict(
        abbr=
        f'OpenHuEval_HuProverbRea_{DATA_VERSION}_OE-prompt_{prompt_template_language}',
        type=HuProverbDatasetOE,
        filepath=DATA_PATH,
        reader_cfg=HuProverbRea_reader_cfg,
        infer_cfg=HuProverbRea_infer_cfg,
        eval_cfg=HuProverbRea_eval_cfg,
    ))
