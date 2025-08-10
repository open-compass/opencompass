from opencompass.openicl.icl_prompt_template import PromptTemplate
from opencompass.openicl.icl_retriever import ZeroRetriever
from opencompass.openicl.icl_inferencer import PPLInferencer
from opencompass.openicl.icl_evaluator import AccEvaluator
from opencompass.datasets import ARCDataset

ARC_c_reader_cfg = dict(
    input_columns=['question', 'textA', 'textB', 'textC', 'textD'],
    output_column='answerKey')

ARC_c_infer_cfg = dict(
    prompt_template=dict(
        type=PromptTemplate,
        template={
            opt: dict(
                round=[
                    dict(role='HUMAN', prompt=f'{{question}}\nA. {{textA}}\nB. {{textB}}\nC. {{textC}}\nD. {{textD}}'),
                    dict(role='BOT', prompt=f'Answer: {opt}'),
                ]
            ) for opt in ['A', 'B', 'C', 'D']
        },
    ),
    retriever=dict(type=ZeroRetriever),
    inferencer=dict(type=PPLInferencer))

ARC_c_eval_cfg = dict(evaluator=dict(type=AccEvaluator))

ARC_c_datasets = [
    dict(
        type=ARCDataset,
        abbr='ARC-c',
        path='opencompass/ai2_arc-dev',
        name='ARC-Challenge',
        reader_cfg=ARC_c_reader_cfg,
        infer_cfg=ARC_c_infer_cfg,
        eval_cfg=ARC_c_eval_cfg)
]
