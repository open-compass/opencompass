from opencompass.openicl.icl_prompt_template import PromptTemplate
from opencompass.openicl.icl_retriever import ZeroRetriever
from opencompass.openicl.icl_inferencer import PPLInferencer
from opencompass.openicl.icl_evaluator import AccEvaluator
from opencompass.datasets import WSCDatasetV3

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
                    prompt='Passage: {text}\nDoes the pronoun # {span2} # refer to * {span1} *?\nA. Yes\nB. No\nAnswer: '
                ),
                dict(role='BOT', prompt='A'),
            ]),
            'B':
            dict(round=[
                dict(
                    role='HUMAN',
                    prompt='Passage: {text}\nDoes the pronoun # {span2} # refer to * {span1} *?\nA. Yes\nB. No\nAnswer: '
                ),
                dict(role='BOT', prompt='B'),
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
        type=WSCDatasetV3,
        path='./data/SuperGLUE/WSC/val.jsonl',
        reader_cfg=WSC_reader_cfg,
        infer_cfg=WSC_infer_cfg,
        eval_cfg=WSC_eval_cfg,
    )
]
