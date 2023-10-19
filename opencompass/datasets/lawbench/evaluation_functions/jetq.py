import re

"""
number prediction
metric: accuracy
金额提取
"""
def compute_jetq(data_dict):
    """
    Compute the Accuracy
    we extract the total amount of cost involved in the crime from the prediction and compare it with the reference
    The prediction is correct if
    the total amount of cost provided in the reference, appears in the prediction.
    """
    score_list, abstentions = [], 0

    for example in data_dict:
        question, prediction, answer = example["origin_prompt"], example["prediction"], example["refr"]
        assert answer.startswith("上文涉及到的犯罪金额:"), f"answer: {answer}, question: {question}"
        assert answer.endswith("元。"), f"answer: {answer}, question: {question}"
        answer = answer.replace("上文涉及到的犯罪金额:", "")

        assert "千元" not in answer, f"answer: {answer}, question: {question}"
        assert "万" not in answer, f"answer: {answer}, question: {question}"

        # remove "元"
        answer = answer.replace("元。", "")
        answer = float(answer)

        prediction_digits = re.findall(r"\d+\.?\d*", prediction)
        prediction_digits = [float(digit) for digit in prediction_digits]

        if len(prediction_digits) == 0:
            abstentions += 1
        if answer in prediction_digits:
            score_list.append(1)
        else:
            score_list.append(0)


    # compute the accuracy of score_list
    accuracy = sum(score_list) / len(score_list)
    return {"score": accuracy, "abstention_rate": abstentions/len(data_dict)}
