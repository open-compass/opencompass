from opencompass.openicl.icl_prompt_template import PromptTemplate
from opencompass.openicl.icl_retriever import ZeroRetriever
from opencompass.openicl.icl_inferencer import PPLInferencer
from opencompass.openicl.icl_evaluator import AccEvaluator
from opencompass.datasets import WSCDatasetV2

WSC_reader_cfg = dict(
    input_columns=['span1', 'span2', 'text'],
    output_column='label',
)

WSC_infer_cfg = dict(
    prompt_template=dict(
        type=PromptTemplate,
        template={
            'A':
            dict(round=[
                dict(
                    role='HUMAN',
                    prompt=
                    "{text}\nIs '{span1}' and '{span2}' refers to the same entity in the above sentence?"
                ),
                dict(role='BOT', prompt='Yes'),
            ]),
            'B':
            dict(round=[
                dict(
                    role='HUMAN',
                    prompt=
                    "{text}\nIs '{span1}' and '{span2}' refers to the same entity in the above sentence?"
                ),
                dict(role='BOT', prompt='No'),
            ]),
        },
    ),
    retriever=dict(type=ZeroRetriever),
    inferencer=dict(type=PPLInferencer),
)

WSC_eval_cfg = dict(evaluator=dict(type=AccEvaluator), )

WSC_datasets = [
    dict(
        abbr='WSC',
        type=WSCDatasetV2,
        path='./data/SuperGLUE/WSC/val.jsonl',
        reader_cfg=WSC_reader_cfg,
        infer_cfg=WSC_infer_cfg,
        eval_cfg=WSC_eval_cfg,
    )
]
