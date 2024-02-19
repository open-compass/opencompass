from opencompass.openicl.icl_prompt_template import PromptTemplate
from opencompass.openicl.icl_retriever import ZeroRetriever
from opencompass.openicl.icl_inferencer import PPLInferencer
from opencompass.openicl.icl_evaluator import AccEvaluator
from opencompass.datasets import WSCDataset

WSC_reader_cfg = dict(
    input_columns=['span1', 'span2', 'text', 'new_text'],
    output_column='answer')

WSC_infer_cfg = dict(
    prompt_template=dict(
        type=PromptTemplate,
        template={
            0: '{text}',
            1: '{new_text}'
        }),
    retriever=dict(type=ZeroRetriever),
    inferencer=dict(type=PPLInferencer))

WSC_eval_cfg = dict(evaluator=dict(type=AccEvaluator))

WSC_datasets = [
    dict(
        type=WSCDataset,
        path='json',
        abbr='WSC',
        data_files='./data/SuperGLUE/WSC/val.jsonl',
        split='train',
        reader_cfg=WSC_reader_cfg,
        infer_cfg=WSC_infer_cfg,
        eval_cfg=WSC_eval_cfg,
    )
]
