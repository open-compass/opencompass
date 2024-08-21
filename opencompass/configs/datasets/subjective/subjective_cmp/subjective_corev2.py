from opencompass.openicl.icl_prompt_template import PromptTemplate
from opencompass.openicl.icl_retriever import ZeroRetriever
from opencompass.openicl.icl_inferencer import GenInferencer
from opencompass.openicl.icl_evaluator import LMEvaluator
from opencompass.datasets import Corev2Dataset
from mmengine.config import read_base

subjective_reader_cfg = dict(
    input_columns=['question', 'prefix', 'suffix'],
    output_column='judge',
    #train_split='test'
    )

subjective_all_sets = [
    'COREV2_6A_all',
]


subjective_datasets = []

for _name in subjective_all_sets:
    subjective_infer_cfg = dict(
            prompt_template=dict(
                type=PromptTemplate,
                template=dict(round=[
                    dict(
                        role='HUMAN',
                        prompt='{question}'
                    ),
                ]),
            ),
            retriever=dict(type=ZeroRetriever),
            inferencer=dict(type=GenInferencer, max_out_len=2048),
        )

    subjective_eval_cfg = dict(
        evaluator=dict(
            type=LMEvaluator,
            infer_order='random',
            prompt_template=dict(
                type=PromptTemplate,
                template=dict(round=[
                    dict(
                        role='HUMAN',
                        prompt = '{prefix}问题: <问题开始> {question} <问题结束>\n\n回答 1: <回答 1 开始> {prediction} <回答 1 结束>\n\n回答 2: <回答 2 开始> {prediction2} <回答 2 结束>\n\n{suffix}'
                    ),
                ]),
            ),
        ),
        pred_role='BOT',
    )

    subjective_datasets.append(
        dict(
            abbr=f'{_name}',
            type=Corev2Dataset,
            path='./data/subjective/',
            name=_name,
            reader_cfg=subjective_reader_cfg,
            infer_cfg=subjective_infer_cfg,
            eval_cfg=subjective_eval_cfg
        ))
