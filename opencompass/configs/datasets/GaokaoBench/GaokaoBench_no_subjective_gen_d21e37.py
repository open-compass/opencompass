import os
from opencompass.openicl.icl_prompt_template import PromptTemplate
from opencompass.openicl.icl_retriever import ZeroRetriever
from opencompass.openicl.icl_inferencer import GenInferencer
from opencompass.datasets import GaokaoBenchDataset
from mmengine.config import read_base

with read_base():
    from .GaokaoBench_prompts import MCQ_prompts, FBQ_prompts

GaokaoBench_datasets = []
for folder, prompts in [
    ('Multiple-choice_Questions', MCQ_prompts),
    ('Fill-in-the-blank_Questions', FBQ_prompts),
]:
    for p in prompts:
        reader_cfg = {
            'input_columns': ['question'],
            'output_column': 'answer',
        }
        infer_cfg = {
            'prompt_template': {
                'type': PromptTemplate,
                'template': p['prefix_prompt'] + '{question}',
            },
            'retriever': {'type': ZeroRetriever},
            'inferencer': {'type': GenInferencer, 'max_out_len': 1024},
        }
        eval_cfg = {
            'evaluator': {'type': 'GaokaoBenchEvaluator' + '_' + p['type']},
            'pred_role': 'BOT',
        }
        dataset = {
            'type': GaokaoBenchDataset,
            'abbr': 'GaokaoBench_' + p['keyword'],
            'path': os.path.join('data', 'GAOKAO-BENCH', 'data', folder, p['keyword'] + '.json'),
            'name': p['keyword'],
            'reader_cfg': reader_cfg,
            'infer_cfg': infer_cfg,
            'eval_cfg': eval_cfg,
        }
        GaokaoBench_datasets.append(dataset)
