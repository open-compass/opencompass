from opencompass.openicl.icl_prompt_template import PromptTemplate
from opencompass.openicl.icl_retriever import ZeroRetriever
from opencompass.openicl.icl_inferencer import PPLInferencer
from opencompass.openicl.icl_evaluator import AccEvaluator
from opencompass.datasets import HFDataset

CB_reader_cfg = dict(
    input_columns=['premise', 'hypothesis'],
    output_column='label',
)

CB_infer_cfg = dict(
    prompt_template=dict(
        type=PromptTemplate,
        template={
            'contradiction':
            dict(round=[
                dict(
                    role='HUMAN',
                    prompt=
                    '{premise}\n{hypothesis}\nWhat is the relation between the two sentences?'
                ),
                dict(role='BOT', prompt='Contradiction'),
            ]),
            'entailment':
            dict(round=[
                dict(
                    role='HUMAN',
                    prompt=
                    '{premise}\n{hypothesis}\nWhat is the relation between the two sentences?'
                ),
                dict(role='BOT', prompt='Entailment'),
            ]),
            'neutral':
            dict(round=[
                dict(
                    role='HUMAN',
                    prompt=
                    '{premise}\n{hypothesis}\nWhat is the relation between the two sentences?'
                ),
                dict(role='BOT', prompt='Neutral'),
            ]),
        },
    ),
    retriever=dict(type=ZeroRetriever),
    inferencer=dict(type=PPLInferencer),
)

CB_eval_cfg = dict(evaluator=dict(type=AccEvaluator), )

CB_datasets = [
    dict(
        type=HFDataset,
        abbr='CB',
        path='json',
        split='train',
        data_files='./data/SuperGLUE/CB/val.jsonl',
        reader_cfg=CB_reader_cfg,
        infer_cfg=CB_infer_cfg,
        eval_cfg=CB_eval_cfg,
    )
]
