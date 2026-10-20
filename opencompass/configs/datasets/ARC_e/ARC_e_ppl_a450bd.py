from opencompass.openicl.icl_prompt_template import PromptTemplate
from opencompass.openicl.icl_retriever import ZeroRetriever
from opencompass.openicl.icl_inferencer import PPLInferencer
from opencompass.openicl.icl_evaluator import AccEvaluator
from opencompass.datasets import ARCDataset

ARC_e_reader_cfg = dict(
    input_columns=['question', 'textA', 'textB', 'textC', 'textD'],
    output_column='answerKey')

ARC_e_infer_cfg = dict(
    prompt_template=dict(
        type=PromptTemplate,
        template={
            'A':
            dict(
                round=[
                    dict(role='HUMAN', prompt='Question: {question}\nAnswer: '),
                    dict(role='BOT', prompt='{textA}')
                ], ),
            'B':
            dict(
                round=[
                    dict(role='HUMAN', prompt='Question: {question}\nAnswer: '),
                    dict(role='BOT', prompt='{textB}')
                ], ),
            'C':
            dict(
                round=[
                    dict(role='HUMAN', prompt='Question: {question}\nAnswer: '),
                    dict(role='BOT', prompt='{textC}')
                ], ),
            'D':
            dict(
                round=[
                    dict(role='HUMAN', prompt='Question: {question}\nAnswer: '),
                    dict(role='BOT', prompt='{textD}')
                ], ),
        }),
    retriever=dict(type=ZeroRetriever),
    inferencer=dict(type=PPLInferencer))

ARC_e_eval_cfg = dict(evaluator=dict(type=AccEvaluator))

ARC_e_datasets = [
    dict(
        type=ARCDataset,
        abbr='ARC-e',
        path='opencompass/ai2_arc-easy-dev',
        name='ARC-Easy',
        reader_cfg=ARC_e_reader_cfg,
        infer_cfg=ARC_e_infer_cfg,
        eval_cfg=ARC_e_eval_cfg)
]
