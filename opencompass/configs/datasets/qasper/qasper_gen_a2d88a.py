from opencompass.openicl.icl_prompt_template import PromptTemplate
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
        type=PromptTemplate,
        template='{evidence}\nAnswer these questions:\nQ: {question}?\nA:'),
    retriever=dict(type=ZeroRetriever),
    inferencer=dict(
        type=GenInferencer, max_out_len=50, max_seq_len=8192, batch_size=4))

qasper_eval_cfg = dict(evaluator=dict(type=TriviaQAEvaluator))

qasper_datasets = [
    dict(
        type=QASPERDataset,
        abbr='QASPER',
        path='./data/QASPER/',
        reader_cfg=qasper_reader_cfg,
        infer_cfg=qasper_infer_cfg,
        eval_cfg=qasper_eval_cfg)
]
