import json
import os

from opencompass.openicl.icl_prompt_template import PromptTemplate
from opencompass.openicl.icl_retriever import ZeroRetriever
from opencompass.openicl.icl_inferencer import GenInferencer
from opencompass.datasets import GaokaoBenchDataset

GaokaoBench_datasets = []
for _folder, _prompt_name in [
    ('Multiple-choice_Questions', 'MCQ_prompt'),
    ('Fill-in-the-blank_Questions', 'FBQ_prompt'),
    ('Open-ended_Questions', 'OEQ_prompt'),
]:
    with open(os.path.join(os.path.dirname(__file__), 'lib_prompt', _prompt_name + '.json'), 'r') as f:
        _prompts = json.load(f)['examples']

    for _p in _prompts:
        _reader_cfg = {
            'input_columns': ['question'],
            'output_column': 'answer',
        }
        _infer_cfg = {
            'ice_template': {
                'type': PromptTemplate,
                'template': {
                    'round': [{
                        'role': 'HUMAN',
                        'prompt': _p['prefix_prompt'] + '{question}'
                    }]
                },
                'ice_token': '</E>'
            },
            'retriever': {
                'type': ZeroRetriever
            },
            'inferencer': {
                'type': GenInferencer,
                'max_out_len': 1024,
            }
        }
        _eval_cfg = {
            'evaluator': {
                'type': 'GaokaoBenchEvaluator' + '_' + _p['type'],
            },
            'pred_role': 'BOT',
        }
        _base_path = './data/GAOKAO-BENCH/data'
        _dataset = {
            'type': GaokaoBenchDataset,
            'abbr': 'GaokaoBench_' + _p['keyword'],
            'path': _base_path + '/' + _folder + '/' + _p['keyword'] + '.json',
            'reader_cfg': _reader_cfg,
            'infer_cfg': _infer_cfg,
            'eval_cfg': _eval_cfg,
        }

        GaokaoBench_datasets.append(_dataset)

_temporary_variables = [k for k in globals() if k.startswith('_')]
for _t in _temporary_variables:
    del globals()[_t]
del _temporary_variables, _t
