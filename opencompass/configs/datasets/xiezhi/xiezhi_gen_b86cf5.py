from opencompass.openicl.icl_prompt_template import PromptTemplate
from opencompass.openicl.icl_inferencer import GenInferencer
from opencompass.openicl.icl_evaluator import AccEvaluator
from opencompass.datasets import XiezhiDataset, XiezhiRetriever
from opencompass.utils.text_postprocessors import first_capital_postprocess

xiezhi_datasets = []

for split in ['spec_eng', 'spec_chn', 'inter_eng', 'inter_chn']:
    if 'chn' in split:
        q_hint, a_hint = '题目', '答案'
    else:
        q_hint, a_hint = 'Question', 'Answer'

    xiezhi_reader_cfg = dict(
        input_columns=['question', 'A', 'B', 'C', 'D', 'labels'],
        output_column='answer',
        train_split='train',
        test_split='test',
    )
    xiezhi_infer_cfg = dict(
        ice_template=dict(
            type=PromptTemplate,
            template=dict(
                begin='</E>',
                round=[
                    dict(role='HUMAN', prompt=f'{q_hint}: {{question}}\nA. {{A}}\nB. {{B}}\nC. {{C}}\nD. {{D}}\n{a_hint}: '),
                    dict(role='BOT', prompt='{answer}'),
                ]
            ),
            ice_token='</E>',
        ),
        retriever=dict(type=XiezhiRetriever, ice_num=3),
        inferencer=dict(type=GenInferencer),
    )

    xiezhi_eval_cfg = dict(evaluator=dict(type=AccEvaluator),
                           pred_role='BOT',
                           pred_postprocessor=dict(type=first_capital_postprocess))

    xiezhi_datasets.append(
        dict(
            type=XiezhiDataset,
            abbr=f'xiezhi-{split}',
            path='./data/xiezhi/',
            name='xiezhi_' + split,
            reader_cfg=xiezhi_reader_cfg,
            infer_cfg=xiezhi_infer_cfg,
            eval_cfg=xiezhi_eval_cfg,
        ))
