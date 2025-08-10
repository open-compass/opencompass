from ..utils.function_utils import multi_choice_judge

"""
multi-choice single-label selection
metric: accuracy
争议焦点：识别案件涉及的争议焦点
"""

def compute_jdzy(data_dict):
    """
    Compute the Accuracy
    The JEC dataset has 16 possible answers for each question, stored in the option_list
    A prediction is correct if
    1. The correct answer appears in the prediction, and
    2. Options other than the answer do not appear in the prediction.
    """

    score_list, abstentions = [], 0
    option_list = ["诉讼主体", "租金情况", "利息", "本金争议", "责任认定", "责任划分", "损失认定及处理",
                   "原审判决是否适当", "合同效力", "财产分割", "责任承担", "鉴定结论采信问题", "诉讼时效", "违约", "合同解除", "肇事逃逸"]
    for example in data_dict:
        question, prediction, answer = example["origin_prompt"], example["prediction"], example["refr"]
        if answer[7:-1] == "赔偿":
            # todo: dataset imperfection
            continue
        assert answer.startswith("争议焦点类别：") and answer[7:-1] in option_list, \
            f"answer: {answer} \n question: {question}"

        answer_letter = answer[7:-1]
        judge = multi_choice_judge(prediction, option_list, answer_letter)
        score_list.append(judge["score"])
        abstentions += judge["abstention"]

    # compute the accuracy of score_list
    accuracy = sum(score_list) / len(score_list)
    return {"score": accuracy, "abstention_rate": abstentions / len(data_dict)}
