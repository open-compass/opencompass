from opencompass.openicl.icl_raw_prompt_template import RawPromptTemplate
from opencompass.openicl.icl_retriever import ZeroRetriever
from opencompass.openicl.icl_inferencer import GenInferencer
from opencompass.datasets import QASPERDataset, TriviaQAEvaluator

qasper_reader_cfg = dict(
    input_columns=['question', 'evidence'],
    output_column='answer',
    train_split='dev',
    test_split='dev')

qasper_infer_cfg = dict(
    prompt_template=dict(
        type=RawPromptTemplate,
        messages=[
            {'role': 'user', 'content': '{evidence}\nAnswer these questions:\nQ: {question}?A:'},
        ],
    ),
    retriever=dict(type=ZeroRetriever),
    inferencer=dict(type=GenInferencer))

qasper_eval_cfg = dict(evaluator=dict(type=TriviaQAEvaluator), pred_role='BOT')

qasper_datasets = [
    dict(
        type=QASPERDataset,
        abbr='QASPER',
        path='opencompass/QASPER',
        reader_cfg=qasper_reader_cfg,
        infer_cfg=qasper_infer_cfg,
        eval_cfg=qasper_eval_cfg)
]
