from ..utils.function_utils import compute_rouge

#法律咨询
def compute_flzx(data_dict):
    """
    Compute the ROUGE-L score between the prediction and the reference
    """
    references, predictions = [], []
    for example in data_dict:
        question, prediction, answer = example["origin_prompt"], example["prediction"], example["refr"]
        predictions.append(prediction)
        references.append(answer)

    # compute the accuracy of score_list
    rouge_scores = compute_rouge(predictions, references)
    rouge_ls = [score["rouge-l"]["f"] for score in rouge_scores]
    average_rouge_l = sum(rouge_ls) / len(rouge_ls)
    return {"score": average_rouge_l}
