from opencompass.openicl.icl_prompt_template import PromptTemplate
from opencompass.openicl.icl_retriever import ZeroRetriever
from opencompass.openicl.icl_inferencer import GenInferencer
from opencompass.datasets import PJExamDataset, PJExamEvaluator

PJExam_datasets = []
for _name in [
        'gk-2022-v1', 'gk-2022-v1-math', 'gk-2023-v1', 'gk-2023-v1-math',
        'gk-2023-v2', 'gk-2023-v2-math', 'zk-2022-v1'
]:
    _hint = '请你做一道</major>选择题\n请你一步一步思考并将思考过程写在【解析】和<eoe>之间。你将从A，B，C，D中选出正确的答案，并写在【答案】和<eoa>之间。\n例如：【答案】A<eoa>\n完整的题目回答的格式如下：\n【解析】...<eoe>\n【答案】...<eoa>\n请你严格按照上述格式作答。\n题目如下：\n'
    _reader_cfg = {
        'input_columns': ['question'],
        'output_column': 'std_ans',
    },
    _infer_cfg = {
        'ice_template': {
            'type': PromptTemplate,
            'template': {
                'round': [{
                    'role': 'HUMAN',
                    'prompt': _hint + '{question}',
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
            'type': PJExamEvaluator
        },
        'pred_role': 'BOT',
        'ds_column': 'eval_infos'
    }
    _dataset = {
        'type': PJExamDataset,
        'abbr': 'PJExamDataset-' + _name,
        'path': './data/PJExam',
        'name': _name,
        'reader_cfg': _reader_cfg,
        'infer_cfg': _infer_cfg,
        'eval_cfg': _eval_cfg,
    }

    PJExam_datasets.append(_dataset)

del _name, _hint, _reader_cfg, _infer_cfg, _eval_cfg, _dataset
