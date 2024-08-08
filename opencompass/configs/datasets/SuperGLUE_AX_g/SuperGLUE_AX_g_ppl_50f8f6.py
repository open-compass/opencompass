from opencompass.openicl.icl_prompt_template import PromptTemplate
from opencompass.openicl.icl_retriever import ZeroRetriever
from opencompass.openicl.icl_inferencer import PPLInferencer
from opencompass.openicl.icl_evaluator import AccEvaluator
from opencompass.datasets import HFDataset

AX_g_reader_cfg = dict(
    input_columns=['hypothesis', 'premise'],
    output_column='label',
    test_split='train')

AX_g_infer_cfg = dict(
    prompt_template=dict(
        type=PromptTemplate,
        template={
            'entailment': '{premise}?entailment, {hypothesis}',
            'not_entailment': '{premise}?not_entailment, {hypothesis}'
        }),
    retriever=dict(type=ZeroRetriever),
    inferencer=dict(type=PPLInferencer))

AX_g_eval_cfg = dict(evaluator=dict(type=AccEvaluator))

AX_g_datasets = [
    dict(
        type=HFDataset,
        abbr='AX_g',
        path='json',
        data_files='./data/SuperGLUE/AX-g/AX-g.jsonl',
        split='train',
        reader_cfg=AX_g_reader_cfg,
        infer_cfg=AX_g_infer_cfg,
        eval_cfg=AX_g_eval_cfg)
]
