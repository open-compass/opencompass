from opencompass.openicl.icl_prompt_template import PromptTemplate
from opencompass.openicl.icl_retriever import ZeroRetriever
from opencompass.openicl.icl_inferencer import PPLInferencer
from opencompass.openicl.icl_evaluator import AccEvaluator
from opencompass.datasets import HFDataset

AX_b_reader_cfg = dict(
    input_columns=['sentence1', 'sentence2'],
    output_column='label',
    test_split='train')

AX_b_infer_cfg = dict(
    prompt_template=dict(
        type=PromptTemplate,
        template={
            'entailment': '{sentence1}?entailment, {sentence2}',
            'not_entailment': '{sentence1}?not_entailment, {sentence2}'
        }),
    retriever=dict(type=ZeroRetriever),
    inferencer=dict(type=PPLInferencer))

AX_b_eval_cfg = dict(evaluator=dict(type=AccEvaluator))

AX_b_datasets = [
    dict(
        type=HFDataset,
        abbr='AX_b',
        path='json',
        data_files='./data/SuperGLUE/AX-b/AX-b.jsonl',
        split='train',
        reader_cfg=AX_b_reader_cfg,
        infer_cfg=AX_b_infer_cfg,
        eval_cfg=AX_b_eval_cfg)
]
