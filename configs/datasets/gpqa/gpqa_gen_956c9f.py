from opencompass.openicl.icl_prompt_template import PromptTemplate
from opencompass.openicl.icl_retriever import ZeroRetriever
from opencompass.openicl.icl_inferencer import GenInferencer
from opencompass.datasets import GPQADataset, gpqa_postprocess, GPQAEvaluator

gpqa_reader_cfg = dict(
    input_columns=['question', 'A', 'B', 'C', 'D'],
    output_column='answer')

gpqa_infer_cfg = dict(
    prompt_template=dict(
        type=PromptTemplate,
        template=dict(
            round=[
                dict(role='HUMAN', prompt='{question}\nChoices:\n'
                                          '(A) {A}\n'
                                          '(B) {B}\n'
                                          '(C) {C}\n'
                                          '(D) {D}\n'
                                          'Format your response as follows: "The correct answer is ( )".'),
            ], )),
    retriever=dict(type=ZeroRetriever),
    inferencer=dict(type=GenInferencer))

gpqa_eval_cfg = dict(evaluator=dict(type=GPQAEvaluator),
                     pred_postprocessor=dict(type=gpqa_postprocess))


gpqa_extended_datasets = [
    dict(
        abbr='GPQA_extended',
        type=GPQADataset,
        path='./data/gpqa/',
        name='gpqa_extended.csv',
        reader_cfg=gpqa_reader_cfg,
        infer_cfg=gpqa_infer_cfg,
        eval_cfg=gpqa_eval_cfg)
]

gpqa_main_datasets = [
    dict(
        abbr='GPQA_main',
        type=GPQADataset,
        path='./data/gpqa/',
        name='gpqa_main.csv',
        reader_cfg=gpqa_reader_cfg,
        infer_cfg=gpqa_infer_cfg,
        eval_cfg=gpqa_eval_cfg)
]

gpqa_diamond_datasets = [
    dict(
        abbr='GPQA_diamond',
        type=GPQADataset,
        path='./data/gpqa/',
        name='gpqa_diamond.csv',
        reader_cfg=gpqa_reader_cfg,
        infer_cfg=gpqa_infer_cfg,
        eval_cfg=gpqa_eval_cfg)
]


