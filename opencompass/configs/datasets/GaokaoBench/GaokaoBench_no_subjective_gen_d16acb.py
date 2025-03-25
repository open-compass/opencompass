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
            'ice_template': {
                'type': PromptTemplate,
                'template': {'round': [{'role': 'HUMAN', 'prompt': p['prefix_prompt'] + '{question}'}]},
                'ice_token': '</E>',
            },
            'retriever': {'type': ZeroRetriever},
            'inferencer': {'type': GenInferencer},
        }
        eval_cfg = {
            'evaluator': {'type': 'GaokaoBenchEvaluator' + '_' + p['type']},
            'pred_role': 'BOT',
        }
        _base_path = 'opencompass/GAOKAO-BENCH'
        dataset = {
            'type': GaokaoBenchDataset,
            'abbr': 'GaokaoBench_' + p['keyword'],
            'path': _base_path,
            'filename': '/' + folder + '/' + p['keyword'] + '.json',
            'name': p['keyword'],
            'reader_cfg': reader_cfg,
            'infer_cfg': infer_cfg,
            'eval_cfg': eval_cfg,
        }
        GaokaoBench_datasets.append(dataset)
