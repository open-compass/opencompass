from rouge_chinese import Rouge
import jieba
from nltk.translate.gleu_score import corpus_gleu

def compute_f1_two_sets(pred_set, gt_set):
    precision = len(pred_set.intersection(gt_set)) / len(pred_set) if len(pred_set) > 0 else 0
    recall = len(pred_set.intersection(gt_set)) / len(gt_set) if len(gt_set) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall > 0 else 0
    return f1

def multi_choice_judge(prediction, option_list, answer_token):
    # a dict, key: letters in the option list, value: count of the letter in the prediction
    count_dict, abstention, accuracy = {}, 0, 0
    for option in option_list:
        option_count = prediction.count(option)
        count_dict[option] = 1 if option_count > 0 else 0  # multiple occurrence of the same letter is counted as 1

    if sum(count_dict.values()) == 0:
        abstention = 1
    # if the answer token is the only predicted token, the prediction is correct 
    elif count_dict[answer_token] == 1 and sum(count_dict.values()) == 1:
        accuracy = 1
    return {"score": accuracy, "abstention": abstention}

"""
compute the rouge score.
hyps and refs are lists of hyposisis and reference strings
empty predictions are replaces with 无内容
"""


def compute_rouge(hyps, refs):
    assert(len(hyps) == len(refs))
    hyps = [' '.join(jieba.cut(h)) for h in hyps]
    hyps = [h if h.strip() != "" else "无内容" for h in hyps]
    refs = [' '.join(jieba.cut(r)) for r in refs]
    return Rouge().get_scores(hyps, refs)

"""
compute the gleu score.
hyps and refs are lists of hyposisis and reference strings
empty predictions are replaces with 无内容
"""
def compute_gleu(hyps, refs):
    assert(len(hyps) == len(refs))
    hyps = [' '.join(jieba.cut(h)) for h in hyps]
    hyps = [h if h.strip() != "" else "无内容" for h in hyps]
    refs = [[' '.join(jieba.cut(r))] for r in refs]
    return corpus_gleu(refs, hyps)
