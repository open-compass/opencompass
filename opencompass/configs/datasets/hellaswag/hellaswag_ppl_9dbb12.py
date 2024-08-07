from opencompass.openicl.icl_prompt_template import PromptTemplate
from opencompass.openicl.icl_retriever import ZeroRetriever
from opencompass.openicl.icl_inferencer import PPLInferencer
from opencompass.openicl.icl_evaluator import AccEvaluator
from opencompass.datasets import HellaswagDataset

hellaswag_reader_cfg = dict(
    input_columns=['ctx', 'A', 'B', 'C', 'D'],
    output_column='label'
)

hellaswag_infer_cfg = dict(
    prompt_template=dict(
        type=PromptTemplate,
        template={
            0: '{ctx} {A}',
            1: '{ctx} {B}',
            2: '{ctx} {C}',
            3: '{ctx} {D}',
        }),
    retriever=dict(type=ZeroRetriever),
    inferencer=dict(type=PPLInferencer))

hellaswag_eval_cfg = dict(evaluator=dict(type=AccEvaluator))

hellaswag_datasets = [
    dict(
        abbr='hellaswag',
        type=HellaswagDataset,
        path='opencompass/hellaswag',
        reader_cfg=hellaswag_reader_cfg,
        infer_cfg=hellaswag_infer_cfg,
        eval_cfg=hellaswag_eval_cfg)
]
