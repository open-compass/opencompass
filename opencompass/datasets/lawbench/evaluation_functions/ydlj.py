from ..utils.comprehension_scores import compute_rc_f1

"""
Task: machine reading comprehension
Metric: F1 score
法律阅读理解
"""
def compute_ydlj(data_dict):
    references, predictions = [], []
    for example in data_dict:
        question, prediction, answer = example["origin_prompt"], example["prediction"], example["refr"]
        answer = answer.replace("回答:", "")
        predictions.append(prediction)
        references.append(answer)

    f1_score = compute_rc_f1(predictions, references)
    return f1_score
