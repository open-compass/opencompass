from opencompass.openicl.icl_prompt_template import PromptTemplate
from opencompass.openicl.icl_retriever import ZeroRetriever
from opencompass.openicl.icl_inferencer import GenInferencer
from opencompass.datasets import QASPERCUTDataset, TriviaQAEvaluator

qaspercut_reader_cfg = dict(
    input_columns=['question', 'evidence'],
    output_column='answer',
    train_split='dev',
    test_split='dev')

qaspercut_infer_cfg = dict(
    prompt_template=dict(
        type=PromptTemplate,
        template='{evidence}\nAnswer these questions:\nQ: {question}?\nA:'),
    retriever=dict(type=ZeroRetriever),
    inferencer=dict(
        type=GenInferencer, max_out_len=50, max_seq_len=8192, batch_size=4))

qaspercut_eval_cfg = dict(evaluator=dict(type=TriviaQAEvaluator))

qaspercut_datasets = [
    dict(
        type=QASPERCUTDataset,
        abbr='qaspercut',
        path='./data/QASPER/',
        reader_cfg=qaspercut_reader_cfg,
        infer_cfg=qaspercut_infer_cfg,
        eval_cfg=qaspercut_eval_cfg)
]
