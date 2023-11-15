from opencompass.openicl.icl_prompt_template import PromptTemplate
from opencompass.openicl.icl_retriever import ZeroRetriever
from opencompass.openicl.icl_inferencer import GenInferencer
from opencompass.openicl.icl_evaluator import EMEvaluator
from opencompass.datasets import XQUADDataset

xquad_reader_cfg = dict(
    input_columns=['question'],
    output_column='answer',
    train_split='test',
    test_split='test')


xquad_infer_cfg = dict(
    prompt_template=dict(
        type=PromptTemplate,
        template=dict(
            round=[
                dict(role='HUMAN', prompt='Read the context and answer the question in the last sentence. Your response should be in Thai and as simple as possible. The context and question is: {question}?'),
                dict(role='BOT', prompt='Answer:'),
            ], )),
    retriever=dict(type=ZeroRetriever),
    inferencer=dict(type=GenInferencer, max_out_len=50))

xquad_eval_cfg = dict(
    evaluator=dict(type=EMEvaluator),
    pred_role="BOT")

xquad_datasets = [
    dict(
        type=XQUADDataset,
        abbr='xquad',
        path='./data/xquad/',
        reader_cfg=xquad_reader_cfg,
        infer_cfg=xquad_infer_cfg,
        eval_cfg=xquad_eval_cfg)
]