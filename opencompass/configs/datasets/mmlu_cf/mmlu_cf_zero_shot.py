from mmengine.config import read_base
from opencompass.openicl.icl_prompt_template import PromptTemplate
from opencompass.openicl.icl_retriever import ZeroRetriever
from opencompass.openicl.icl_inferencer import GenInferencer
from opencompass.openicl.icl_evaluator import AccwithDetailsEvaluator
from opencompass.datasets import MMLUCFDataset
from opencompass.utils.text_postprocessors import first_option_postprocess

with read_base():
    from .mmlu_cf_categories import categories

mmlu_cf_reader_cfg = dict(
    input_columns=['input', 'A', 'B', 'C', 'D'],
    output_column='target',
    train_split='dev')

mmlu_cf_datasets = []
for _name in categories:
    _hint = f'There is a single choice question (with answers). Answer the question by replying A, B, C or D.'
    mmlu_cf_infer_cfg = dict(
        ice_template=dict(
            type=PromptTemplate,
            template=dict(round=[
                dict(
                    role='HUMAN',
                    prompt=
                    f'{_hint}\nQuestion: {{input}}\nA. {{A}}\nB. {{B}}\nC. {{C}}\nD. {{D}}\nAnswer: '
                ),
                dict(role='BOT', prompt='{target}\n')
            ]),
        ),
        prompt_template=dict(
            type=PromptTemplate,
            template=dict(
                begin='</E>',
                round=[
                    dict(
                        role='HUMAN',
                        prompt=f'{_hint}\nQuestion: {{input}}\nA. {{A}}\nB. {{B}}\nC. {{C}}\nD. {{D}}\nAnswer: '
                    ),
                ],
            ),
            ice_token='</E>',
        ),
        retriever=dict(type=ZeroRetriever),
        inferencer=dict(type=GenInferencer),
    )

    mmlu_cf_eval_cfg = dict(
        evaluator=dict(type=AccwithDetailsEvaluator),
        pred_postprocessor=dict(type=first_option_postprocess, options='ABCD'))

    mmlu_cf_datasets.append(
        dict(
            abbr=f'mmlu_cf_{_name}',
            type=MMLUCFDataset,
            path='microsoft/MMLU-CF',
            name=_name,
            reader_cfg=mmlu_cf_reader_cfg,
            infer_cfg=mmlu_cf_infer_cfg,
            eval_cfg=mmlu_cf_eval_cfg,
        ))

del _name, _hint
