def compute_acc(gt_list, pred_list):
    correct_num = 0
    for pred, gold in zip(pred_list, gt_list):
        kept_pred = round(pred, 4) if (pred is not None) else pred
        kept_gold = round(gold, 4)
        if kept_pred == kept_gold:
            correct_num += 1
    acc = correct_num / len(gt_list)
    return acc
