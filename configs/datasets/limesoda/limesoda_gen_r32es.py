from opencompass.openicl.icl_prompt_template import PromptTemplate
from opencompass.openicl.icl_retriever import ZeroRetriever
from opencompass.openicl.icl_inferencer import GenInferencer
from opencompass.openicl.icl_evaluator import EMEvaluator
from opencompass.datasets import LimesodaDataset


limesoda_reader_cfg = dict(
    input_columns=['document'],
    output_column='label',
    test_split='test')

limesoda_infer_cfg = dict(
    prompt_template=dict(
        type=PromptTemplate,
        template=dict(
            round=[
                dict(role='HUMAN', prompt='Read the given document in Thai, label the document using one of these labels: ("Fact News", "Fake News", or "Undefined"). Document: {document}?'),
                dict(role='BOT', prompt='Label:'),
            ], )),
    retriever=dict(type=ZeroRetriever),
    inferencer=dict(type=GenInferencer, max_out_len=50))

limesoda_eval_cfg = dict(
    evaluator=dict(type=EMEvaluator),
    pred_role="BOT")


limesoda_datasets = [
    dict(
        type=LimesodaDataset,
        abbr='limesoda',
        path='./data/limesoda/',
        reader_cfg=limesoda_reader_cfg,
        infer_cfg=limesoda_infer_cfg,
        eval_cfg=limesoda_eval_cfg)
]


