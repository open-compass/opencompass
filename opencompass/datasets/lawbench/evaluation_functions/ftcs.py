from ..utils.function_utils import compute_rouge

#法条记忆问答
def compute_ftcs(data_dict):
    """
    Compute the ROUGE-L score between the prediction and the reference
    """
    references, predictions = [], []
    for example in data_dict:
        question, prediction, answer = example["origin_prompt"], example["prediction"], example["refr"]
        answer = answer.replace("答案:", "")
        predictions.append(prediction)
        references.append(answer)

    # compute the accuracy of score_list
    rouge_scores = compute_rouge(predictions, references)
    rouge_ls = [score["rouge-l"]["f"] for score in rouge_scores]
    average_rouge_l = sum(rouge_ls) / len(rouge_ls)
    return {"score": average_rouge_l}
