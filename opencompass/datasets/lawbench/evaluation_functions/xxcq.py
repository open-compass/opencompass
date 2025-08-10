from ..utils.comprehension_scores import compute_ie_f1


"""
task: information extraction
metric: F1 score
信息抽取
"""
def compute_xxcq(data_dict):
    references, predictions = [], []
    for example in data_dict:
        question, prediction, answer = example["origin_prompt"], example["prediction"], example["refr"]
        predictions.append(prediction)
        references.append(answer)

    return compute_ie_f1(predictions, references, {"犯罪嫌疑人", "受害人", "被盗货币", "物品价值", "盗窃获利",
                                                   "被盗物品", "作案工具", "时间", "地点", "组织机构"})
