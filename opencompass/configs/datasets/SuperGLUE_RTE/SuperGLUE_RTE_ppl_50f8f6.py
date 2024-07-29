from opencompass.openicl.icl_prompt_template import PromptTemplate
from opencompass.openicl.icl_retriever import ZeroRetriever
from opencompass.openicl.icl_inferencer import PPLInferencer
from opencompass.openicl.icl_evaluator import AccEvaluator
from opencompass.datasets import HFDataset

RTE_reader_cfg = dict(
    input_columns=['hypothesis', 'premise'],
    output_column='label',
    test_split='train')

RTE_infer_cfg = dict(
    prompt_template=dict(
        type=PromptTemplate,
        template={
            'entailment': '{premise}?entailment, {hypothesis}',
            'not_entailment': '{premise}?not_entailment, {hypothesis}'
        }),
    retriever=dict(type=ZeroRetriever),
    inferencer=dict(type=PPLInferencer))

RTE_eval_cfg = dict(evaluator=dict(type=AccEvaluator))

RTE_datasets = [
    dict(
        type=HFDataset,
        abbr='RTE',
        path='json',
        data_files='./data/SuperGLUE/RTE/val.jsonl',
        split='train',
        reader_cfg=RTE_reader_cfg,
        infer_cfg=RTE_infer_cfg,
        eval_cfg=RTE_eval_cfg)
]
