def compute_acc(gt_list, pred_list):
    correct_num = sum(pred == gt for gt, pred in zip(gt_list, pred_list))
    acc = correct_num / len(gt_list)
    return acc
