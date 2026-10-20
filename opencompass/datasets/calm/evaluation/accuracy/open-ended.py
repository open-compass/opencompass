import jieba
from rouge import Rouge


def is_chinese(text):
    for char in text:
        if '\u4e00' <= char <= '\u9fff':
            return True
    return False


def compute_acc(gt_list, pred_list):
    rouge_l = 0
    rouge = Rouge()

    for pred, gold in zip(pred_list, gt_list):
        if is_chinese(pred):
            prediction = ' '.join(jieba.cut(pred))
            gold = ' '.join(jieba.cut(gold))
        else:
            prediction = pred
            gold = gold

        try:
            scores = rouge.get_scores(prediction, gold)
            rouge_l += scores[0]['rouge-l']['r']
        except Exception as e:
            print(f'copmute rouge_l error occurred: {e}')
            continue
    avg_rougel = rouge_l / len(gt_list)
    return avg_rougel
