from opencompass.openicl.icl_prompt_template import PromptTemplate
from opencompass.openicl.icl_retriever import ZeroRetriever
from opencompass.openicl.icl_inferencer import GenInferencer
from opencompass.datasets import KaoshiDataset

prompts = {
    
        "单选题" : "请你做一道单项选择题\n请你一步一步思考并将思考过程写在【解析】和<eoe>之间。你将从A，B，C，D中选出正确的答案，并写在【答案】和<eoa>之间，答案应只包含最终结果，不要添加额外词语。\n例如：【答案】: A <eoa>\n完整的题目回答的格式如下：\n【解析】 ... <eoe>\n【答案】 ... <eoa>\n请你严格按照上述格式作答。\n题目如下：",
        "多选题" : "请你做一道多项选择题\n请你一步一步思考并将思考过程写在【解析】和<eoe>之间。你将从A，B，C，D中选出正确的答案，答案可能是一个到多个选项，奇怪将其写在【答案】和<eoa>之间，答案应只包含最终结果，不要添加额外词语。\n例如：【答案】: A D <eoa>\n完整的题目回答的格式如下：\n【解析】 ... <eoe>\n【答案】 ... <eoa>\n请你严格按照上述格式作答。\n题目如下：",
        # "解答题" : "请解答下面的解答题\n仔细阅读题目并充分结合你已有的知识，解答其中的问题，请你一步步思考并将思考过程写在【解析】和<eoe>之间。你的答案请写在【答案】和<eoa>之间\n完整的题目回答格式如下：\n(1)【解析】 ...<eoe>\n【答案】...<eoa>\n (2)【解析】 ...<eoe>\n【答案】...<eoa>\n请你严格按照上述格式作答，如果不止一道题，请分别作答。\n题目如下：" ,
        "填空题":"请解答下面的填空题\n仔细阅读题目，解答其中的问题，请你一步步思考并将思考过程写在【解析】和<eoe>之间。请把你的答案写在【答案】和<eoa>之间，答案应只包含最终结果，不要添加额外词语。\n完整的题目回答格式如下：\n【解析】 ...<eoe>\n【答案】...<eoa>\n请你严格按照上述格式作答。\n题目如下:",
}


# splits = ['考研-经济', '职业-公务员', '考研-法学', '职业-高项', '职业-出版', '职业-测绘', '考研-数学', '考研-管理类综合', '职业-基金', '职业-银行', '职业-会计', '职业-建筑', '职业-消防', '职业-教师资格', '职业-期货', '考研-英语', '职业-房地产估价', '考研-临床医学', '考研-政治', '职业-安全工程', '职业-证券']
splits_with_type = {'单选题': ['职业-消防', '职业-测绘', '考研-经济', '职业-安全工程', '考研-政治', '职业-建筑', '考研-英语', '职业-教师资格', '职业-证券', '职业-会计', '职业-公务员', '考研-数学', '考研-法学', '职业-高项', '考研-临床医学', '职业-银行', '考研-管理类综合', '职业-基金'], 
                    '多选题': ['职业-消防', '职业-测绘', '考研-政治', '职业-建筑', '职业-证券', '职业-会计', '考研-法学', '考研-临床医学', '职业-银行'], 
                    '完形填空': ['考研-英语'], 
                    '判断题': ['职业-证券'], 
                    '填空题': ['考研-数学']}
Kaoshi_datasets = []

# for _folder, _prompts in [
#     ("Multiple-choice_Questions", _MCQ_prompts),
#     ("Fill-in-the-blank_Questions", _FBQ_prompts),
#     ("Open-ended_Questions", _OEQ_prompts),
# ]:

for _type in ['单选题', '多选题', '填空题']:
    for _split in splits_with_type[_type]:
        if "法学" in _split or "房地产" in _split:
            continue
        _folder = _split.replace('-' + _type, '')
        _p = prompts[_type]
        _reader_cfg = {
            "input_columns": ['question'],
            "output_column": 'answer',
        }
        _infer_cfg = {
            "ice_template": {
                "type": PromptTemplate,
                "template": {
                    "round": [{
                        "role": "HUMAN",
                        "prompt": _p + '{question}'
                    }]
                },
                "ice_token": "</E>"
            },
            "retriever": {
                "type": ZeroRetriever
            },
            "inferencer": {
                "type": GenInferencer,
                "max_out_len": 1024,
            }
        }
        _eval_cfg = {
            "evaluator": {
                "type": "KaoshiEvaluator" + "_" + _type,
            },
            "pred_role": "BOT",
        }
        _base_path = './data/Kaoshi'
        _dataset = {
            "type": KaoshiDataset,
            "abbr": "Kaoshi" + _split,
            "path": _base_path + '/' + _folder + '/' + _type + ".jsonl",
            "reader_cfg": _reader_cfg,
            "infer_cfg": _infer_cfg,
            "eval_cfg": _eval_cfg,
        }

        Kaoshi_datasets.append(_dataset)

_temporary_variables = [k for k in globals() if k.startswith('_')]
for _t in _temporary_variables:
    del globals()[_t]
del _temporary_variables, _t
