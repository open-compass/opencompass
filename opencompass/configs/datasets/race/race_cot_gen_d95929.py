from opencompass.openicl.icl_prompt_template import PromptTemplate
from opencompass.openicl.icl_retriever import ZeroRetriever
from opencompass.openicl.icl_inferencer import GenInferencer
from opencompass.openicl.icl_evaluator import AccEvaluator
from opencompass.datasets import RaceDataset
from opencompass.utils.text_postprocessors import (
    first_option_postprocess,
)

QUERY_TEMPLATE = """
Answer the following multiple choice question. The last line of your response should be of the following format: 'ANSWER: $LETTER' (without quotes) where LETTER is one of ABCD. Think step by step before answering.

Article: {article}

Q: {question}

A. {A}
B. {B}
C. {C}
D. {D}
""".strip()

race_reader_cfg = dict(
    input_columns=['article', 'question', 'A', 'B', 'C', 'D'],
    output_column='answer',
    train_split='validation',
    test_split='test',
)

race_infer_cfg = dict(
    prompt_template=dict(
        type=PromptTemplate,
        template=dict(
            round=[
                dict(role='HUMAN', prompt=QUERY_TEMPLATE),
            ]
        ),
    ),
    retriever=dict(type=ZeroRetriever),
    inferencer=dict(type=GenInferencer),
)

race_eval_cfg = dict(
    evaluator=dict(type=AccEvaluator),
    pred_postprocessor=dict(type=first_option_postprocess, options='ABCD'),
    pred_role='BOT',
)

race_datasets = [
    dict(
        abbr='race-middle',
        type=RaceDataset,
        path='opencompass/race',
        name='middle',
        reader_cfg=race_reader_cfg,
        infer_cfg=race_infer_cfg,
        eval_cfg=race_eval_cfg,
    ),
    dict(
        abbr='race-high',
        type=RaceDataset,
        path='opencompass/race',
        name='high',
        reader_cfg=race_reader_cfg,
        infer_cfg=race_infer_cfg,
        eval_cfg=race_eval_cfg,
    ),
]
