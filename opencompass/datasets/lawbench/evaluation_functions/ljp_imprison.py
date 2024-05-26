import math

import re

#法律判决预测-刑期预测
def compute_ljp_imprison(data_dict):
    import cn2an

    score_list, abstentions = [], 0

    for example in data_dict:
        question, prediction, answer = example["origin_prompt"], example["prediction"], example["refr"]
        # get answer digit, which is the number between "刑期:" and "个月"
        if "死刑" in answer or "无期" in answer:
            # TODO: data imperfection
            continue

        assert answer.startswith("刑期:") and answer.endswith("个月"), f"answer: {answer}, question: {question}"
        answer = answer.replace("刑期:", "")
        answer = answer.replace("个月", "")
        answer_digit = int(answer)
        prediction = cn2an.transform(prediction, "cn2an")

        # use regular expression to extract the digits from prediction, only consider digits before "个月" or "月"
        prediction_digit_month_list = re.findall(r"\d+个月", prediction)
        prediction_digit_month_list = [int(digit.replace("个月", "")) for digit in prediction_digit_month_list]
        prediction_digit_month_list2 = re.findall(r"\d+月", prediction)
        prediction_digit_month_list2 = [int(digit.replace("月", "")) for digit in prediction_digit_month_list2]
        prediction_digit_month_list.extend(prediction_digit_month_list2)
        # catches the digits before "年"
        prediction_digit_year_list = re.findall(r"\d+年", prediction)
        prediction_digit_year_list = [int(digit.replace("年", "")) for digit in prediction_digit_year_list]

        if len(prediction_digit_month_list) > 0:
            prediction_digit_month = int(prediction_digit_month_list[0])
        elif len(prediction_digit_year_list) > 0:
            prediction_digit_month = int(prediction_digit_year_list[0]) * 12
        else:
            abstentions += 1
            prediction_digit_month = -1

        if prediction_digit_month != -1:
            score_list.append(abs(math.log(answer_digit + 1) - math.log(prediction_digit_month + 1)))
        else:
            score_list.append(math.log(216))

    # compute the average of score_list (log distance)
    log_distance = sum(score_list) / len(score_list)
    # normalize the score to between 0 and 1
    log_distance = (math.log(216) - log_distance)/math.log(216)
    return {"score": log_distance, "abstention_rate": abstentions/len(data_dict)}
