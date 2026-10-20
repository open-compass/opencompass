from opencompass.openicl.icl_prompt_template import PromptTemplate
from opencompass.openicl.icl_retriever import ZeroRetriever, FixKRetriever
from opencompass.openicl.icl_evaluator import AccEvaluator
from opencompass.datasets import BoolQDatasetV2
from opencompass.openicl.icl_inferencer import PPLInferencer
from opencompass.openicl.icl_evaluator import AccEvaluator

BoolQ_reader_cfg = dict(
    input_columns=['question', 'passage'],
    output_column='label',
)

BoolQ_infer_cfg = dict(
    ice_template=dict(
        type=PromptTemplate,
        template={
            'B': dict(
                round=[
                    dict(role='HUMAN', prompt='{passage}\nQuestion: {question}?'),
                    dict(role='BOT', prompt='No'),
                ]
            ),
            'A': dict(
                round=[
                    dict(role='HUMAN', prompt='{passage}\nQuestion: {question}?'),
                    dict(role='BOT', prompt='Yes'),
                ]
            ),
        },
        ice_token='</E>',
    ),
    retriever=dict(type=FixKRetriever, fix_id_list=[0, 2, 4, 6, 8]),
    inferencer=dict(type=PPLInferencer, max_out_len=50),
)

BoolQ_eval_cfg = dict(evaluator=dict(type=AccEvaluator))

BoolQ_datasets = [
    dict(
        abbr='BoolQ',
        type=BoolQDatasetV2,
        path='opencompass/boolq',
        reader_cfg=BoolQ_reader_cfg,
        infer_cfg=BoolQ_infer_cfg,
        eval_cfg=BoolQ_eval_cfg,
    )
]
