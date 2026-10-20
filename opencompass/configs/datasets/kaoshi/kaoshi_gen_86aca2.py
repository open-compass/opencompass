from opencompass.datasets import KaoshiDataset, KaoshiEvaluator
from opencompass.openicl.icl_inferencer import GenInferencer
from opencompass.openicl.icl_prompt_template import PromptTemplate
from opencompass.openicl.icl_retriever import ZeroRetriever

prompts = {
        '单选题' : '请你做一道单项选择题\n请你一步一步思考并将思考过程写在【解析】和<eoe>之间。你将从A，B，C，D中选出正确的答案，并写在【答案】和<eoa>之间，答案应只包含最终结果，不要添加额外词语。\n例如：【答案】: A <eoa>\n完整的题目回答的格式如下：\n【解析】 ... <eoe>\n【答案】 ... <eoa>\n请你严格按照上述格式作答。\n题目如下：',
        '多选题' : '请你做一道多项选择题\n请你一步一步思考并将思考过程写在【解析】和<eoe>之间。你将从多个选项中选出正确的答案，答案可能是一个到多个选项，奇怪将其写在【答案】和<eoa>之间，答案应只包含最终结果，不要添加额外词语。\n例如：【答案】: A D <eoa>\n完整的题目回答的格式如下：\n【解析】 ... <eoe>\n【答案】 ... <eoa>\n请你严格按照上述格式作答。\n题目如下：',
        '填空题' : '请解答下面的填空题\n仔细阅读题目，解答其中的问题，请你一步步思考并将思考过程写在【解析】和<eoe>之间。请把你的答案写在【答案】和<eoa>之间，答案应只包含最终结果，不要添加额外词语。\n完整的题目回答格式如下：\n【解析】 ... <eoe>\n【答案】... <eoa>\n请你严格按照上述格式作答。\n题目如下:',
        '完形填空' : '请你做一道英语完形填空题,其中包含二十个小题。\n请你一步一步思考。每一题你将从A，B，C，D中选出正确的答案，并写在【答案】和<eoa>之间。\n例如：（1）【答案】 A <eoa>\n（2）【答案】 B <eoa>\n请你严格按照上述格式作答。\n',
        '七选五': '请回答下面的问题，将符合题意的五个选项的字母写在【答案】和<eoa>之间，例如：【答案】 A B C D E <eoa>\n请严格按照上述格式作答。题目如下：\n',
        '判断题' : '请回答下面的判断题，将你的判断结果写在【答案】和<eoa>之间，若给定表述正确时回答：\n【答案】正确 <eoa>\n 表述错误时回答：\n【答案】错误 <eoa>\n请严格按照上述格式作答。题目如下：\n',
}

splits_with_type = {'单选题': ['职业-消防', '职业-测绘', '考研-经济', '职业-安全工程', '考研-政治', '职业-建筑', '考研-英语', '职业-教师资格', '职业-证券', '职业-会计', '职业-公务员', '考研-数学', '职业-高项', '考研-临床医学', '职业-银行', '考研-管理类综合', '职业-基金'],
                    '多选题': ['职业-消防', '职业-测绘', '考研-政治', '职业-建筑', '职业-证券', '职业-会计', '考研-临床医学', '职业-银行'],
                    '完形填空': ['考研-英语'],
                    '七选五': ['考研-英语'],
                    '判断题': ['职业-证券'],
                    '填空题': ['考研-数学']}

zh2en = {'单选题': 'single_choice', '多选题': 'multi_choice', '完形填空': 'multi_question_choice', '判断题': 'judgment', '填空题': 'cloze', '七选五': 'five_out_of_seven'}

kaoshi_datasets = []

for _type in list(splits_with_type.keys()):
    for _split in splits_with_type[_type]:
        _folder = _split.replace('-' + _type, '')
        _p = prompts[_type]
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
                        'prompt': _p + '{question}'
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
                'type': KaoshiEvaluator,
                  'question_type': zh2en[_type],
            },
            'pred_role': 'BOT',
        }
        _base_path = './data/Kaoshi'
        _dataset = {
            'type': KaoshiDataset,
            'abbr': 'Kaoshi' + _split + '-' + _type,
            'path': _base_path + '/' + _folder + '/' + _type + '.jsonl',
            'name': zh2en[_type],
            'reader_cfg': _reader_cfg,
            'infer_cfg': _infer_cfg,
            'eval_cfg': _eval_cfg,
        }

        kaoshi_datasets.append(_dataset)

_temporary_variables = [k for k in globals() if k.startswith('_')]
for _t in _temporary_variables:
    del globals()[_t]
del _temporary_variables, _t
