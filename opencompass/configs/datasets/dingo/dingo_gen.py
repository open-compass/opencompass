from opencompass.openicl.icl_prompt_template import PromptTemplate
from opencompass.openicl.icl_retriever import ZeroRetriever
from opencompass.openicl.icl_inferencer import GenInferencer
from opencompass.datasets import DingoDataset, DingoEvaluator


dingo_paths = [
    './data/dingo/en_192.csv',
    './data/dingo/zh_170.csv',
]

dingo_datasets = []
for path in dingo_paths:
    dingo_reader_cfg = dict(input_columns='input', output_column=None)
    dingo_infer_cfg = dict(
        prompt_template=dict(
            type=PromptTemplate,
            template=dict(round=[dict(role='HUMAN', prompt='{input}')])),
        retriever=dict(type=ZeroRetriever),
        inferencer=dict(type=GenInferencer),
    )
    dingo_eval_cfg = dict(evaluator=dict(type=DingoEvaluator), pred_role='BOT')

    dingo_datasets.append(
        dict(
            abbr='dingo_' + path.split('/')[-1].split('.csv')[0],
            type=DingoDataset,
            path=path,
            reader_cfg=dingo_reader_cfg,
            infer_cfg=dingo_infer_cfg,
            eval_cfg=dingo_eval_cfg,
        ))

datasets = dingo_datasets
