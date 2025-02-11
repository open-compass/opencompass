from mmengine.config import read_base

from opencompass.datasets.OpenHuEval.HuSimpleQA import HuSimpleQADataset
from opencompass.openicl.icl_evaluator import LMEvaluator
from opencompass.openicl.icl_inferencer import GenInferencer
from opencompass.openicl.icl_prompt_template import PromptTemplate
from opencompass.openicl.icl_retriever import ZeroRetriever

with read_base():
    from .HuSimpleQA_setting import DATA_PATH, DATA_VERSION, INSTRUCTIONS

PROMPT_LANGUAGES = ['en', 'hu']

HuSimpleQA_reader_cfg = dict(
    input_columns=['question', 'prompt'],
    output_column='references',
)

HuSimpleQA_datasets = []
for lang in PROMPT_LANGUAGES:
    instruction = INSTRUCTIONS[lang]

    HuSimpleQA_infer_cfg = dict(
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
        inferencer=dict(
            type=GenInferencer,
            max_seq_len=8192,
            max_out_len=8192,
        ),
    )

    HuSimpleQA_eval_cfg = dict(
        evaluator=dict(
            type=LMEvaluator,
            prompt_template=dict(
                type=PromptTemplate,
                template="""{prompt}"""
            ),
        ),
        pred_role='BOT',
    )

    HuSimpleQA_datasets.append(
        dict(
            abbr=f'OpenHuEval_HuSimpleQA_{DATA_VERSION}-prompt_{lang}',
            type=HuSimpleQADataset,
            filepath=DATA_PATH,
            reader_cfg=HuSimpleQA_reader_cfg,
            infer_cfg=HuSimpleQA_infer_cfg,
            eval_cfg=HuSimpleQA_eval_cfg,
            mode='singlescore'
        ))
