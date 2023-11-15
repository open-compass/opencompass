from opencompass.openicl.icl_prompt_template import PromptTemplate
from opencompass.openicl.icl_retriever import ZeroRetriever
from opencompass.openicl.icl_inferencer import GenInferencer
from opencompass.openicl.icl_evaluator import EMEvaluator
from opencompass.datasets import WiseSentiDataset


wisesenti_reader_cfg = dict(
    input_columns=['message'],
    output_column='label',
    test_split='test')


wisesenti_infer_cfg = dict(
    prompt_template=dict(
        type=PromptTemplate,
        template=dict(
            round=[
                dict(role='HUMAN', prompt='Read the given message in Thai, label the sentiment of the message using one of these labels: (positive, neutral, negative, question). Message: {message}?'),
                dict(role='BOT', prompt='Label:'),
            ], )),
    retriever=dict(type=ZeroRetriever),
    inferencer=dict(type=GenInferencer, max_out_len=50))

wisesenti_eval_cfg = dict(
    evaluator=dict(type=EMEvaluator),
    pred_role="BOT")

wisesenti_datasets = [
    dict(
        type=WiseSentiDataset,
        abbr='wisesenti',
        path='./data/wisesight_senti/',
        reader_cfg=wisesenti_reader_cfg,
        infer_cfg=wisesenti_infer_cfg,
        eval_cfg=wisesenti_eval_cfg)
]

#  