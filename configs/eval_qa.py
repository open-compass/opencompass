from mmengine.config import read_base

from opencompass.openicl.icl_prompt_template import PromptTemplate
from opencompass.openicl.icl_retriever import ZeroRetriever
from opencompass.openicl.icl_inferencer import GenInferencer
from opencompass.datasets import qaDataset, qaEvaluator


with read_base():
    from .models.hf_internlm.hf_internlm_7b import models

qa_paths = [
    './data/qa/en_162.csv',
    './data/qa/zh_90.csv',
]

qa_datasets = []
for path in qa_paths:
    qa_reader_cfg = dict(input_columns='input', output_column=None)
    qa_infer_cfg = dict(
        prompt_template=dict(
            type=PromptTemplate,
            template=dict(round=[dict(role='HUMAN', prompt='{input}')])),
        retriever=dict(type=ZeroRetriever),
        inferencer=dict(type=GenInferencer),
    )
    qa_eval_cfg = dict(evaluator=dict(type=qaEvaluator), pred_role='BOT')

    qa_datasets.append(
        dict(
            abbr='qa_' + path.split('/')[-1].split('.csv')[0],
            type=qaDataset,
            path=path,
            reader_cfg=qa_reader_cfg,
            infer_cfg=qa_infer_cfg,
            eval_cfg=qa_eval_cfg,
        ))

datasets = qa_datasets

work_dir = './outputs/eval_qa'
