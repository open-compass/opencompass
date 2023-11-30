from ..utils.function_utils import compute_f1_two_sets
from ..utils.rc_f1 import CJRCEvaluator


"""
task: event detection
metric: F1 score
事件检测
"""
option_list = ["支付/给付", "欺骗", "搜查/扣押", "要求/请求", "卖出", "买入", "获利", "拘捕", "鉴定", "同意/接受", "供述", "联络", "帮助/救助", "租用/借用", "受伤", "伪造", "卖淫", "伤害人身", "赔偿", "归还/偿还"]

def compute_sjjc(data_dict):
    """
    Compute the F1-score
    The sjjc task covers 20 event types.
    A question may involve one or more event types.
    Given a list of event types from both the ground truth and the prediction, we compute the F1-score between
    these two lists.
    """
    score_list, abstentions = [], 0

    for example in data_dict:
        question, prediction, answer = example["origin_prompt"], example["prediction"], example["refr"]

        answers = answer.split(";")

        prediction_list =[]
        for option in option_list:
            if option in prediction:
                prediction_list.append(option)

        if len(prediction_list) == 0:
            abstentions += 1
        gt_set = set(answers)
        pred_set = set(prediction_list)
        score = compute_f1_two_sets(gt_set, pred_set)
        score_list.append(score)

    f1_score_average = sum(score_list) / len(score_list)
    return {"score": f1_score_average, "abstention_rate": abstentions/len(data_dict)}

"""
task: trigger word extraction
metric: F1 score
触发词抽取
"""
def compute_cfcy(data_dict):

    scores = 0

    for example in data_dict:
        question, prediction, answer = example["origin_prompt"], example["prediction"], example["refr"]

        answers = answer.split(";")
        predictions = prediction.split(";")
        intersected = [CJRCEvaluator.compute_f1(r, h) for r, h in zip(answers, predictions)]

        prec = sum(intersected) / len(predictions) if len(predictions) > 0 else 0
        rec = sum(intersected) / len(answers) if len(answers) > 0 else 0
        # print(prec, rec, intersected)
        scores += 2 * prec * rec / (prec + rec + 1e-10)

    f1_score_average = scores / len(data_dict)
    return {"score": f1_score_average}
