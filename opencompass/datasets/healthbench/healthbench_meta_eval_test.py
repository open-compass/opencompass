from . import healthbench_meta_eval


def test_compute_agreement_for_rater_by_class():
    self_pred_list = [True, False, True]
    other_preds_list = [[True, True, False], [True, False], [False]]
    cluster_list = ['a', 'a', 'b']
    model_or_physician = 'model'
    metrics = healthbench_meta_eval.compute_metrics_for_rater_by_class(
        self_pred_list, other_preds_list, cluster_list, model_or_physician
    )

    # precision overall
    index_str_pos_precision = healthbench_meta_eval.INDEX_STR_TEMPLATE.format(
        model_or_physician=model_or_physician, metric='precision', pred_str='pos'
    )
    index_str_neg_precision = healthbench_meta_eval.INDEX_STR_TEMPLATE.format(
        model_or_physician=model_or_physician, metric='precision', pred_str='neg'
    )
    overall_pos_precision = metrics[index_str_pos_precision]
    overall_neg_precision = metrics[index_str_neg_precision]
    expected_overall_pos_precision = (2 + 0 + 0) / (3 + 0 + 1)
    expected_overall_neg_precision = (0 + 1 + 0) / (0 + 2 + 0)
    assert overall_pos_precision['value'] == expected_overall_pos_precision
    assert overall_neg_precision['value'] == expected_overall_neg_precision
    assert overall_pos_precision['n'] == 4
    assert overall_neg_precision['n'] == 2

    # recall overall
    index_str_pos_recall = healthbench_meta_eval.INDEX_STR_TEMPLATE.format(
        model_or_physician=model_or_physician, metric='recall', pred_str='pos'
    )
    index_str_neg_recall = healthbench_meta_eval.INDEX_STR_TEMPLATE.format(
        model_or_physician=model_or_physician, metric='recall', pred_str='neg'
    )
    overall_pos_recall = metrics[index_str_pos_recall]
    overall_neg_recall = metrics[index_str_neg_recall]
    expected_overall_pos_recall = (2 + 0 + 0) / (2 + 1 + 0)
    expected_overall_neg_recall = (0 + 1 + 0) / (1 + 1 + 1)
    assert overall_pos_recall['value'] == expected_overall_pos_recall
    assert overall_neg_recall['value'] == expected_overall_neg_recall
    assert overall_pos_recall['n'] == 3
    assert overall_neg_recall['n'] == 3

    # f1 overall
    index_str_pos_f1 = healthbench_meta_eval.INDEX_STR_TEMPLATE.format(
        model_or_physician=model_or_physician, metric='f1', pred_str='pos'
    )
    index_str_neg_f1 = healthbench_meta_eval.INDEX_STR_TEMPLATE.format(
        model_or_physician=model_or_physician, metric='f1', pred_str='neg'
    )
    overall_pos_f1 = metrics[index_str_pos_f1]
    overall_neg_f1 = metrics[index_str_neg_f1]
    expected_overall_pos_f1 = (
        2
        * expected_overall_pos_precision
        * expected_overall_pos_recall
        / (expected_overall_pos_precision + expected_overall_pos_recall)
    )
    expected_overall_neg_f1 = (
        2
        * expected_overall_neg_precision
        * expected_overall_neg_recall
        / (expected_overall_neg_precision + expected_overall_neg_recall)
    )
    assert overall_pos_f1['value'] == expected_overall_pos_f1
    assert overall_neg_f1['value'] == expected_overall_neg_f1

    # balanced f1
    index_str_balanced_f1 = healthbench_meta_eval.INDEX_STR_TEMPLATE.format(
        model_or_physician=model_or_physician, metric='f1', pred_str='balanced'
    )
    balanced_f1 = metrics[index_str_balanced_f1]
    expected_balanced_f1 = (expected_overall_pos_f1 + expected_overall_neg_f1) / 2
    assert balanced_f1['value'] == expected_balanced_f1

    # by cluster
    # precision
    cluster_a_str_pos_precision = healthbench_meta_eval.CLUSTER_STR_TEMPLATE.format(
        cluster='a', index_str=index_str_pos_precision
    )
    cluster_a_str_neg_precision = healthbench_meta_eval.CLUSTER_STR_TEMPLATE.format(
        cluster='a', index_str=index_str_neg_precision
    )
    cluster_a_pos_precision = metrics[cluster_a_str_pos_precision]
    cluster_a_neg_precision = metrics[cluster_a_str_neg_precision]
    assert cluster_a_pos_precision['value'] == (
        # example 1, 2 in order
        (2 + 0) / (3 + 0)
    )
    assert cluster_a_neg_precision['value'] == (
        # example 1, 2 in order
        (0 + 1) / (0 + 2)
    )
    assert cluster_a_pos_precision['n'] == 3
    assert cluster_a_neg_precision['n'] == 2

    # recall
    cluster_a_str_pos_recall = healthbench_meta_eval.CLUSTER_STR_TEMPLATE.format(
        cluster='a', index_str=index_str_pos_recall
    )
    cluster_a_str_neg_recall = healthbench_meta_eval.CLUSTER_STR_TEMPLATE.format(
        cluster='a', index_str=index_str_neg_recall
    )
    cluster_a_pos_recall = metrics[cluster_a_str_pos_recall]
    cluster_a_neg_recall = metrics[cluster_a_str_neg_recall]
    assert cluster_a_pos_recall['value'] == (
        # example 1, 2 in order
        (2 + 0) / (2 + 1)
    )
    assert cluster_a_neg_recall['value'] == (
        # example 1, 2 in order
        (0 + 1) / (1 + 1)
    )
    assert cluster_a_pos_recall['n'] == 3
    assert cluster_a_neg_recall['n'] == 2

    # cluster B
    # precision
    cluster_b_str_pos_precision = healthbench_meta_eval.CLUSTER_STR_TEMPLATE.format(
        cluster='b', index_str=index_str_pos_precision
    )
    cluster_b_str_neg_precision = healthbench_meta_eval.CLUSTER_STR_TEMPLATE.format(
        cluster='b', index_str=index_str_neg_precision
    )
    cluster_b_str_pos_precision = metrics[cluster_b_str_pos_precision]
    assert cluster_b_str_neg_precision not in metrics
    assert cluster_b_str_pos_precision['value'] == (
        # example 3 only
        0 / 1
    )
    assert cluster_b_str_pos_precision['n'] == 1

    # recall
    cluster_b_str_pos_recall = healthbench_meta_eval.CLUSTER_STR_TEMPLATE.format(
        cluster='b', index_str=index_str_pos_recall
    )
    cluster_b_str_neg_recall = healthbench_meta_eval.CLUSTER_STR_TEMPLATE.format(
        cluster='b', index_str=index_str_neg_recall
    )
    assert cluster_b_str_pos_recall not in metrics
    cluster_b_neg_recall = metrics[cluster_b_str_neg_recall]
    assert cluster_b_neg_recall['value'] == (
        # example 3 only
        0 / 1
    )
    assert cluster_b_neg_recall['n'] == 1

    # f1
    index_str_pos_f1 = healthbench_meta_eval.CLUSTER_STR_TEMPLATE.format(
        cluster='b', index_str=index_str_pos_f1
    )
    index_str_neg_f1 = healthbench_meta_eval.CLUSTER_STR_TEMPLATE.format(
        cluster='b', index_str=index_str_neg_f1
    )
    index_str_balanced_f1 = healthbench_meta_eval.CLUSTER_STR_TEMPLATE.format(
        cluster='b', index_str=index_str_balanced_f1
    )
    assert index_str_pos_f1 not in metrics
    assert index_str_neg_f1 not in metrics
    assert index_str_balanced_f1 not in metrics


if __name__ == '__main__':
    test_compute_agreement_for_rater_by_class()
