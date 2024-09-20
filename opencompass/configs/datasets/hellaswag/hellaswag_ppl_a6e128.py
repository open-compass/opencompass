from opencompass.openicl.icl_prompt_template import PromptTemplate
from opencompass.openicl.icl_retriever import ZeroRetriever
from opencompass.openicl.icl_inferencer import PPLInferencer
from opencompass.openicl.icl_evaluator import AccEvaluator
from opencompass.datasets import HellaswagDataset_V3

hellaswag_reader_cfg = dict(
    input_columns=['query', 'A', 'B', 'C', 'D'],
    output_column='gold')

hellaswag_infer_cfg = dict(
    prompt_template=dict(
        type=PromptTemplate,
        template={
            '0': dict(
                round=[dict(role='HUMAN', prompt='{query} {A}')]
            ),
            '1': dict(
                round=[dict(role='HUMAN', prompt='{query} {B}')]
            ),
            '2': dict(
                round=[dict(role='HUMAN', prompt='{query} {C}')]
            ),
            '3': dict(
                round=[dict(role='HUMAN', prompt='{query} {D}')]
            ),
        }),
    retriever=dict(type=ZeroRetriever),
    inferencer=dict(type=PPLInferencer))

hellaswag_eval_cfg = dict(evaluator=dict(type=AccEvaluator))

hellaswag_datasets = [
    dict(
        abbr='hellaswag',
        type=HellaswagDataset_V3,
        path='opencompass/hellaswag',
        reader_cfg=hellaswag_reader_cfg,
        infer_cfg=hellaswag_infer_cfg,
        eval_cfg=hellaswag_eval_cfg)
]
