from opencompass.datasets import CFinBench, CFinBenchEvaluator
from opencompass.openicl.icl_inferencer import GenInferencer
from opencompass.openicl.icl_inferencer import PPLInferencer
from opencompass.openicl.icl_prompt_template import PromptTemplate
from opencompass.openicl.icl_retriever import ZeroRetriever
from opencompass.openicl.icl_retriever import FixKRetriever

prompts = {
    "单选题": "请你做一道关于{}的单项选择题，\n你将从A，B，C，D中选出正确的答案，并写在'答案：'之后，答案应只包含最终结果，不要添加额外语句。\n例如：答案： B。请你严格按照上述格式作答。\n题目如下：",
    "多选题": "请你做一道关于{}的多项选择题，\n答案可能是一个到多个选项,请你从四个或者五个选项中选出正确的答案，并将其写在'答案：'之后，答案应只包含最终结果，不要添加额外语句。\n例如：答案: A D。请你严格按照上述格式作答。\n题目如下：",
    "判断题": "请回答下面关于{}的判断题，将你的判断结果写在'答案：'之后，答案应只包含最终结果，不要添加额外语句。若给定题目表述正确，则回答：'答案：正确'，否则回答：'答案：错误'。请严格按照上述格式作答。题目如下：",
}

NUM_1 = 11
NUM_2 = 8
NUM_3 = 13
NUM_4 = 11
splits_with_type = {'单选题': [f"1-{i + 1}" for i in range(NUM_1)]
                    + [f"2-{i + 1}" for i in range(NUM_2)]
                    + [f"3-{i + 1}" for i in range(NUM_3)]
                    + [f"4-{i + 1}" for i in range(NUM_4)],
                    }

mapping = {"1-1": "政治经济学",
           "1-2": "西方经济学",
           "1-3": "微观经济学",
           "1-4": "宏观经济学",
           "1-5": "产业经济学",
           "1-6": "财政学",
           "1-7": "国际贸易学",
           "1-8": "统计学",
           "1-9": "审计学",
           "1-10": "经济史",
           "1-11": "金融学",
           "2-1": "税务从业资格",
           "2-2": "期货从业资格",
           "2-3": "基金从业资格",
           "2-4": "地产从业资格",
           "2-5": "保险从业资格",
           "2-6": "证券从业资格",
           "2-7": "银行从业资格",
           "2-8": "注册会计师",
           "3-1": "初级审计师",
           "3-2": "中级审计师",
           "3-3": "初级统计师",
           "3-4": "中级统计师",
           "3-5": "初级经济师",
           "3-6": "中级经济师",
           "3-7": "初级银行从业人员",
           "3-8": "中级银行从业人员",
           "3-9": "初级会计师",
           "3-10": "中级会计师",
           "3-11": "税务师",
           "3-12": "资产评估师",
           "3-13": "证券分析师",
           "4-1": "税法一",
           "4-2": "税法二",
           "4-3": "税务稽查",
           "4-4": "商业法",
           "4-5": "证券法",
           "4-6": "保险法",
           "4-7": "经济法",
           "4-8": "银行业法",
           "4-9": "期货法",
           "4-10": "金融法",
           "4-11": "民法"}

zh2en = {'单选题': 'single_choice', '多选题': 'multi_choice', '判断题': 'judgment'}

cfinbench_datasets = []

for _set in ['val']:
    for _type in list(splits_with_type.keys()):
        # _type：单选题、多选题、判断题
        for _split in splits_with_type[_type]:
            # _split：1-1、2-1
            _p = prompts[_type].format(mapping[_split])

            _reader_cfg = dict(
                input_columns='text',
                output_column='Answer',
                train_split="dev",
                test_split=_set
            )

            text1 = "{text}\n答案: "
            _infer_cfg = dict(
                prompt_template=dict(
                    type=PromptTemplate,
                    template={answer: rf"{_p}\n{text1}{answer}" for answer in ["A", "B", "C", "D"]},
                ),

                retriever=dict(
                    type=ZeroRetriever
                ),
                inferencer=dict(
                    type=PPLInferencer
                )
            )

            _eval_cfg = dict(
                evaluator=dict(
                    type=CFinBenchEvaluator,
                    data_type=zh2en[_type]
                ),
                pred_role="BOT"
            )
            _base_path = './data/CFinBench'
            _dataset = dict(
                abbr="CFinBench" + _set + _split + '-' + 'ppl-' + _type + '_zero-shot',
                type=CFinBench,
                path=_base_path,
                name=_split + ".jsonl",
                data_type=zh2en[_type],
                reader_cfg=_reader_cfg,
                infer_cfg=_infer_cfg,
                eval_cfg=_eval_cfg
            )

            cfinbench_datasets.append(_dataset)

_temporary_variables = [k for k in globals() if k.startswith('_')]
for _t in _temporary_variables:
    del globals()[_t]
del _temporary_variables, _t
