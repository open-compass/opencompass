"""
task: multiple choice classification
metric: F1 score
婚姻文本分类
"""

def compute_wbfl(data_dict):
    """
    A reference (R) contains a list of options, each option is from the option_list.
    We will extract the options appearing in the prediction and convert them into a set (P).
    We compute the F1 score between the prediction (P) and the reference (R).
    """


    score_list, abstentions = [], 0
    option_list = ["婚后有子女", "限制行为能力子女抚养", "有夫妻共同财产", "支付抚养费", "不动产分割", "婚后分局",
                   "二次起诉离婚", "按月给付抚养费", "准予离婚", "有夫妻共同债务", "婚前个人财产", "法定离婚", "不履行家庭义务",
                   "存在非婚生子", "适当帮助", "不履行离婚协议", "损害赔偿", "感情不和分居满二年", "子女随非抚养权人生活", "婚后个人财产"]
    for example in data_dict:
        question, prediction, answer = example["origin_prompt"], example["prediction"], example["refr"]
        assert answer.startswith("类别:") and answer.endswith("。"), f"answer: {answer}, question: {question}"

        gt_list = (answer[3:-1].split("、"))
        for gt in gt_list:
            assert gt in option_list, f"gt: {gt}, question: {question}"
        gt_set = set(gt_list)

        prediction_list = []
        for option in option_list:
            if option in prediction:
                prediction_list.append(option)
        if len(prediction_list) == 0:
            abstentions += 1
        predict_set = set(prediction_list)
        precision = len(gt_set.intersection(predict_set)) / len(predict_set) if len(predict_set) != 0 else 0
        recall = len(gt_set.intersection(predict_set)) / len(gt_set) if len(gt_set) != 0 else 0
        f1_score = 2 * precision * recall / (precision + recall) if (precision + recall) != 0 else 0
        score_list.append(f1_score)

    # compute the accuracy of score_list
    final_f1_score = sum(score_list) / len(score_list)
    return {'score': final_f1_score, 'abstention_rate': abstentions / len(data_dict)}
