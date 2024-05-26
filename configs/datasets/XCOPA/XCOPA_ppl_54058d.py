from opencompass.openicl.icl_prompt_template import PromptTemplate
from opencompass.openicl.icl_retriever import ZeroRetriever
from opencompass.openicl.icl_inferencer import PPLInferencer
from opencompass.openicl.icl_evaluator import AccEvaluator
from opencompass.datasets import XCOPADataset

XCOPA_reader_cfg = dict(
    input_columns=['question', 'premise', 'choice1', 'choice2'],
    output_column='label',
    test_split='train')

XCOPA_infer_cfg = dict(
    prompt_template=dict(
        type=PromptTemplate,
        template={
            0: 'Premise:{premise}。\nQuestion:{question}。\nAnswer: {choice1}.',
            1: 'Passage:{premise}。\nQuestion:{question}。\nAnswer: {choice2}.',
        }),
    retriever=dict(type=ZeroRetriever),
    inferencer=dict(type=PPLInferencer))

XCOPA_eval_cfg = dict(evaluator=dict(type=AccEvaluator))

XCOPA_datasets = [
    dict(
        type=XCOPADataset,
        path='xcopa',
        reader_cfg=XCOPA_reader_cfg,
        infer_cfg=XCOPA_infer_cfg,
        eval_cfg=XCOPA_eval_cfg)
]
