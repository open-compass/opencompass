import re

"""
task: law article prediction
metric: F1 score
法律判决预测-法条预测
"""
def replace_match(match):
    return match.group(1)

def compute_ljp_article(data_dict):
    """
    Compute the F1-score
    A reference contains a list of articles of the Criminal Law of the People's Republic of China.
    We compute the F1-score between the prediction and the reference.
    """
    import cn2an

    score_list, abstentions = [], 0

    for example in data_dict:
        question, prediction, answer = example["origin_prompt"], example["prediction"], example["refr"]
        assert answer.startswith("法条:刑法第"), f"answer: {answer}"
        assert answer.endswith("条"), f"answer: {answer}"

        answer = answer.replace("法条:刑法第", "")
        answer = answer.replace("条", "")

        answer_law_indices = answer.split("、")
        answer_law_index_digit_list = []
        for answer_law_index in answer_law_indices:
            assert answer_law_index.isdigit(), f"answer_law_index: {answer_law_index}"
            answer_law_index_digit = int(answer_law_index)
            assert answer_law_index_digit <= 490, "刑法总共只有490条"
            answer_law_index_digit_list.append(answer_law_index_digit)

        prediction_law_chunks = prediction.split("、")
        prediction_law_index_digit_list = []

        for prediction_law_chunk in prediction_law_chunks:
            prediction_law_chunk = prediction_law_chunk.replace("万元", "元")

            # delete phrase starts with "第" and ends with "款", we don't care about it in the answer
            prediction_law_chunk = re.sub(r'第(.*?)款', "", prediction_law_chunk)
            # keep only the digits in the phrase starts with "第" and ends with "条", otherwise cn may fail to convert
            prediction_law_chunk = re.sub(r'第(.*?)条', replace_match, prediction_law_chunk)
            prediction_law_chunk = cn2an.transform(prediction_law_chunk, "cn2an")
            # find digtis in prediction_law_chunk
            prediction_law_section_numbers = re.findall(r"\d+", prediction_law_chunk)
            if len(prediction_law_section_numbers) == 0:
                continue
            if len(prediction_law_section_numbers) != 1:
                # in this case, we only take the first number, and reject the others
                pass

            prediction_law_index_digit = int(prediction_law_section_numbers[0])
            prediction_law_index_digit_list.append(prediction_law_index_digit)

        gt_set = set(answer_law_index_digit_list)
        pred_set = set(prediction_law_index_digit_list)
        if len(pred_set) == 0:
            abstentions += 1
        precision = len(gt_set.intersection(pred_set)) / len(pred_set) if len(pred_set) != 0 else 0
        recall = len(gt_set.intersection(pred_set)) / len(gt_set) if len(gt_set) != 0 else 0
        f1_score = 2 * precision * recall / (precision + recall) if (precision + recall) != 0 else 0
        score_list.append(f1_score)

    # compute the accuracy of score_list
    average_f1 = sum(score_list) / len(score_list)
    return {'score': average_f1, 'abstention_rate': abstentions/len(data_dict)}
