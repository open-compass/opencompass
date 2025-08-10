from ..utils.function_utils import multi_choice_judge

"""
Task: multi-choice selection
Metric: Accuracy
论辩挖掘
"""
def compute_lblj(data_dict):
    """
    Compute the Accuracy
    The LBLJ dataset has 5 options for each question: A, B, C, D, E
    A prediction is correct if
    1. The correct answer appears in the prediction, and
    2. Options other than the answer do not appear in the prediction.
    """
    score_list, abstentions = [], 0
    option_list = ["A", "B", "C", "D", "E"]
    for example in data_dict:
        question, prediction, answer = example["origin_prompt"], example["prediction"], example["refr"]
        assert answer.startswith("[正确答案]") and answer[6] in option_list, f"answer[6]: {answer}, question: {question}"

        answer_letter = answer[6]
        judge = multi_choice_judge(prediction, option_list, answer_letter)
        score_list.append(judge["score"])
        abstentions += judge["abstention"]

    # compute the accuracy of score_list
    accuracy = sum(score_list) / len(score_list)
    return {"score": accuracy, "abstention_rate": abstentions / len(data_dict)}
