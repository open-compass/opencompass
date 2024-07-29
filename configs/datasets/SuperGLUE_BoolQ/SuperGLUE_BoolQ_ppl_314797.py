from opencompass.openicl.icl_prompt_template import PromptTemplate
from opencompass.openicl.icl_retriever import ZeroRetriever
from opencompass.openicl.icl_inferencer import PPLInferencer
from opencompass.openicl.icl_evaluator import AccEvaluator
from opencompass.datasets import BoolQDatasetV3

BoolQ_reader_cfg = dict(
    input_columns=['question', 'passage'],
    output_column='label',
    test_split='train')

BoolQ_infer_cfg = dict(
    prompt_template=dict(
        type=PromptTemplate,
        template={
            'false':
            dict(round=[
                dict(role='HUMAN', prompt='Passage: {passage}\nQuestion: {question}?'),
                dict(role='BOT', prompt='Answer: No'),
            ]),
            'true':
            dict(round=[
                dict(role='HUMAN', prompt='Passage: {passage}\nQuestion: {question}?'),
                dict(role='BOT', prompt='Answer: Yes'),
            ]),
        },
    ),
    retriever=dict(type=ZeroRetriever),
    inferencer=dict(type=PPLInferencer),
)

BoolQ_eval_cfg = dict(evaluator=dict(type=AccEvaluator))

BoolQ_datasets = [
    dict(
        abbr='BoolQ',
        type=BoolQDatasetV3,
        path='./data/SuperGLUE/BoolQ/val.jsonl',
        reader_cfg=BoolQ_reader_cfg,
        infer_cfg=BoolQ_infer_cfg,
        eval_cfg=BoolQ_eval_cfg,
    )
]
