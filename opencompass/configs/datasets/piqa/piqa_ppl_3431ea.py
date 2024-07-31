from opencompass.openicl.icl_prompt_template import PromptTemplate
from opencompass.openicl.icl_retriever import ZeroRetriever
from opencompass.openicl.icl_inferencer import PPLInferencer
from opencompass.openicl.icl_evaluator import AccEvaluator
from opencompass.datasets import PIQADataset

piqa_reader_cfg = dict(
    input_columns=['goal', 'sol1', 'sol2'],
    output_column='label',
    test_split='validation')

piqa_infer_cfg = dict(
    prompt_template=dict(
        type=PromptTemplate,
        template={
            0:
            dict(
                round=[
                    dict(role='HUMAN', prompt='{goal}'),
                    dict(role='BOT', prompt='{sol1}')
                ], ),
            1:
            dict(
                round=[
                    dict(role='HUMAN', prompt='{goal}'),
                    dict(role='BOT', prompt='{sol2}')
                ], )
        }),
    retriever=dict(type=ZeroRetriever),
    inferencer=dict(type=PPLInferencer))

piqa_eval_cfg = dict(evaluator=dict(type=AccEvaluator))

piqa_datasets = [
    dict(
        abbr='piqa',
        type=PIQADataset,
        path='opencompass/piqa',
        reader_cfg=piqa_reader_cfg,
        infer_cfg=piqa_infer_cfg,
        eval_cfg=piqa_eval_cfg)
]
