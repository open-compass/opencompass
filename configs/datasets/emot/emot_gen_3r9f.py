from opencompass.openicl.icl_prompt_template import PromptTemplate
from opencompass.openicl.icl_retriever import ZeroRetriever
from opencompass.openicl.icl_inferencer import GenInferencer
from opencompass.openicl.icl_evaluator import EMEvaluator
from opencompass.datasets import EmotDataset



emot_reader_cfg = dict(
    input_columns=['text'],
    output_column='label',
    test_split='test')

emot_infer_cfg = dict(
    prompt_template=dict(
        type=PromptTemplate,
        template=dict(
            round=[
                dict(role='HUMAN', prompt='Read the provided Indonesian text and assign one of the following emotion labels: anger, fear, happiness, love, and sadness. Text: {text}?'),
                dict(role='BOT', prompt='Label:'),
            ], )),
    retriever=dict(type=ZeroRetriever),
    inferencer=dict(type=GenInferencer, max_out_len=50))

emot_eval_cfg = dict(
    evaluator=dict(type=EMEvaluator),
    pred_role="BOT")


emot_datasets = [
    dict(
        type=EmotDataset,
        abbr='emot',
        path='./data/emot/',
        reader_cfg=emot_reader_cfg,
        infer_cfg=emot_infer_cfg,
        eval_cfg=emot_eval_cfg)
]


