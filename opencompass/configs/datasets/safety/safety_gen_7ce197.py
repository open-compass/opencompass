from opencompass.openicl.icl_prompt_template import PromptTemplate
from opencompass.openicl.icl_retriever import ZeroRetriever
from opencompass.openicl.icl_inferencer import GenInferencer
from opencompass.openicl.icl_evaluator import ToxicEvaluator
from opencompass.datasets import SafetyDataset

safety_reader_cfg = dict(
    input_columns=['prompt'],
    output_column='idx',
    train_split='test',
    test_split='test')

# TODO: allow empty output-column
safety_infer_cfg = dict(
    prompt_template=dict(
        type=PromptTemplate,
        template='{prompt}'),
    retriever=dict(type=ZeroRetriever),
    inferencer=dict(type=GenInferencer))

safety_eval_cfg = dict(evaluator=dict(type=ToxicEvaluator), )

safety_datasets = [
    dict(
        type=SafetyDataset,
        path='./data/safety.txt',
        reader_cfg=safety_reader_cfg,
        infer_cfg=safety_infer_cfg,
        eval_cfg=safety_eval_cfg)
]
