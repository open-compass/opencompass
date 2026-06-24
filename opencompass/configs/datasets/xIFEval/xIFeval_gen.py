from opencompass.openicl.icl_prompt_template import PromptTemplate
from opencompass.openicl.icl_retriever import ZeroRetriever
from opencompass.openicl.icl_inferencer import GenInferencer
from opencompass.datasets import xIFEvalDataset, xIFEvaluator

xifeval_reader_cfg = dict(
    input_columns=['prompt'], output_column='reference')

xifeval_infer_cfg = dict(
    prompt_template=dict(
        type=PromptTemplate,
        template=dict(round=[
            dict(
                role='HUMAN',
                prompt='{prompt}'),
        ])),
    retriever=dict(type=ZeroRetriever),
    inferencer=dict(type=GenInferencer, max_out_len=4096))

xifeval_eval_cfg = dict(
    evaluator=dict(type=xIFEvaluator),
    pred_role='BOT',
)

xifeval_datasets = []

LANGS = ['ar', 'bn', 'cs', 'de', 'es', 'fr', 'hu', 'ja', 'ko', 'ru', 'sr', 'sw', 'te', 'th', 'vi', 'zh']
for LANG in LANGS:
    path = f'data/xifeval/input_data_google_{LANG}.jsonl'
    xifeval_datasets.append(dict(
        type=xIFEvalDataset,
        abbr=f'xIFEval_{LANG}',
        path=path,
        reader_cfg=xifeval_reader_cfg,
        infer_cfg=xifeval_infer_cfg,
        eval_cfg=xifeval_eval_cfg,
    ))
