# flake8: noqa: E501
import importlib
import json
from pathlib import Path

task_to_accuracy_module_map = {
    # association/
    # correlation/
    'CORR-B_correlation_CN': 'choice',
    'CORR-B_correlation_EN': 'choice',
    # explaining_away_effect/
    'EAE-B_exp-away_CN': 'choice',
    'EAE-B_exp-away_EN': 'choice',
    # causal_discovery/
    # abstract_reasoning/
    'AR-B_CaLM-AR_CN': 'choice',
    'AR-B_CaLM-AR_EN': 'choice',
    # causal_attribution/
    'CA-B_FA_CN': 'choice',
    'CA-B_FA_EN': 'choice',
    'CA-B_FP_CN': 'choice',
    'CA-B_FP_EN': 'choice',
    # event_causality_identification/
    'ECI-B_CTB_CN': 'choice',
    'ECI-B_CTB_EN': 'choice',
    'ECI-B_ESC_CN': 'choice',
    'ECI-B_ESC_EN': 'choice',
    'ECI-B_MAVEN-ERE_CN': 'choice',
    'ECI-B_MAVEN-ERE_EN': 'choice',
    # pairwise_causal_discovery/
    'PCD-B_COPA_CN': 'choice',
    'PCD-B_COPA_EN': 'choice',
    'PCD-B_E-CARE_CN': 'choice',
    'PCD-B_E-CARE_EN': 'choice',
    'PCD-C_COPA_CN': 'choice',
    'PCD-C_COPA_EN': 'choice',
    'PCD-C_E-CARE_CN': 'choice',
    'PCD-C_E-CARE_EN': 'choice',
    # counterfactual/
    # actual_causality/
    'AC-B_causal_judgement_CN': 'choice',
    'AC-B_causal_judgement_EN': 'choice',
    # causal_explanation_generation/
    'CEG-O_E-CARE_CN': 'open-ended',
    'CEG-O_E-CARE_EN': 'open-ended',
    # counterfactual_reasoning/
    'CR-B_det-counterfactual_CN': 'choice',
    'CR-B_det-counterfactual_EN': 'choice',
    'CR-C_CRASS_CN': 'choice',
    'CR-C_CRASS_EN': 'choice',
    # effect_of_the_treatment_on_the_treated/
    'ETT-B_ETT-natural_CN': 'choice',
    'ETT-B_ETT-natural_EN': 'choice',
    'ETT-P_ETT-basic_CN': 'prob',
    'ETT-P_ETT-basic_EN': 'prob',
    'ETT-P_ETT-hard_CN': 'prob',
    'ETT-P_ETT-hard_EN': 'prob',
    # natural_direct_effect/
    'NDE-B_NDE-natural_CN': 'choice',
    'NDE-B_NDE-natural_EN': 'choice',
    'NDE-P_NDE-basic_CN': 'prob',
    'NDE-P_NDE-basic_EN': 'prob',
    'NDE-P_NDE-hard_CN': 'prob',
    'NDE-P_NDE-hard_EN': 'prob',
    # natural_indirect_effect/
    'NIE-B_NIE-natural_CN': 'choice',
    'NIE-B_NIE-natural_EN': 'choice',
    'NIE-P_NIE-basic_CN': 'prob',
    'NIE-P_NIE-basic_EN': 'prob',
    'NIE-P_NIE-hard_CN': 'prob',
    'NIE-P_NIE-hard_EN': 'prob',
    # probability_of_necessity/
    'PN-P_PN-basic_CN': 'prob',
    'PN-P_PN-basic_EN': 'prob',
    'PN-P_PN-hard_CN': 'prob',
    'PN-P_PN-hard_EN': 'prob',
    # probability_of_sufficiency/
    'PS-P_PS-basic_CN': 'prob',
    'PS-P_PS-basic_EN': 'prob',
    'PS-P_PS-hard_CN': 'prob',
    'PS-P_PS-hard_EN': 'prob',
    # intervention/
    # average_treatment_effect/
    'ATE-B_ATE-natural_CN': 'choice',
    'ATE-B_ATE-natural_EN': 'choice',
    'ATE-P_ATE-basic_CN': 'prob',
    'ATE-P_ATE-basic_EN': 'prob',
    'ATE-P_ATE-hard_CN': 'prob',
    'ATE-P_ATE-hard_EN': 'prob',
    # backdoor_adjustment_set/
    'BAS-B_backadj_CN': 'choice',
    'BAS-B_backadj_EN': 'choice',
    'BAS-C_max-BAS_CN': 'choice',
    'BAS-C_max-BAS_EN': 'choice',
    'BAS-C_min-BAS_CN': 'choice',
    'BAS-C_min-BAS_EN': 'choice',
    'BAS-C_mix-BAS_CN': 'choice',
    'BAS-C_mix-BAS_EN': 'choice',
    # causal_effect_identification/
    'CEI-B_0.2-UC_CN': 'choice',
    'CEI-B_0.2-UC_EN': 'choice',
    'CEI-B_0.4-UC_CN': 'choice',
    'CEI-B_0.4-UC_EN': 'choice',
    'CEI-B_0.6-UC_CN': 'choice',
    'CEI-B_0.6-UC_EN': 'choice',
    'CEI-B_0.8-UC_CN': 'choice',
    'CEI-B_0.8-UC_EN': 'choice',
    # collider_bias/
    'CB-B_collider-bias_CN': 'choice',
    'CB-B_collider-bias_EN': 'choice',
    # controlled_direct_effect/
    'CDE-B_CDE-natural_CN': 'choice',
    'CDE-B_CDE-natural_EN': 'choice',
    'CDE-P_CDE-basic_CN': 'prob',
    'CDE-P_CDE-basic_EN': 'prob',
    'CDE-P_CDE-hard_CN': 'prob',
    'CDE-P_CDE-hard_EN': 'prob',
    # frontdoor_adjustment_set/
    'FAS-C_FAS_CN': 'choice',
    'FAS-C_FAS_EN': 'choice',
    # instrumental_variable/
    'IV-C_CaLM-IV_CN': 'choice',
    'IV-C_CaLM-IV_EN': 'choice',
}


def initialize_core_metric_evaluation_components(task):
    """Loads the labeling and accuracy functions dynamically based on the
    specified task for core metric computation.

    Parameters:
    - task: The specific task to load functions for.

    Returns:
    - Tuple containing the ground truth labeling function, prediction labeling function,
      and the accuracy function.

    Raises:
    - NotImplementedError: If no functions are found for the specified task.
    """
    task_to_labeling_module_map = {
        # association/
        # correlation/
        'CORR-B_correlation_CN': 'CLADDER',
        'CORR-B_correlation_EN': 'CLADDER',
        # explaining_away_effect/
        'EAE-B_exp-away_CN': 'CLADDER',
        'EAE-B_exp-away_EN': 'CLADDER',
        # causal_discovery/
        # abstract_reasoning/
        'AR-B_CaLM-AR_CN': 'AR-B_CaLM-AR',
        'AR-B_CaLM-AR_EN': 'AR-B_CaLM-AR',
        # causal_attribution/
        'CA-B_FA_CN': 'CA-B_FA',
        'CA-B_FA_EN': 'CA-B_FA',
        'CA-B_FP_CN': 'CA-B_FP',
        'CA-B_FP_EN': 'CA-B_FP',
        # event_causality_identification/
        'ECI-B_CTB_CN': 'ECI',
        'ECI-B_CTB_EN': 'ECI',
        'ECI-B_ESC_CN': 'ECI',
        'ECI-B_ESC_EN': 'ECI',
        'ECI-B_MAVEN-ERE_CN': 'ECI',
        'ECI-B_MAVEN-ERE_EN': 'ECI',
        # pairwise_causal_discovery/
        'PCD-B_COPA_CN': 'PCD-B',
        'PCD-B_COPA_EN': 'PCD-B',
        'PCD-B_E-CARE_CN': 'PCD-B',
        'PCD-B_E-CARE_EN': 'PCD-B',
        'PCD-C_COPA_CN': 'PCD-C',
        'PCD-C_COPA_EN': 'PCD-C',
        'PCD-C_E-CARE_CN': 'PCD-C',
        'PCD-C_E-CARE_EN': 'PCD-C',
        # counterfactual/
        # actual_causality/
        'AC-B_causal_judgement_CN': 'AC-B_causal_judgement',
        'AC-B_causal_judgement_EN': 'AC-B_causal_judgement',
        # causal_explanation_generation/
        'CEG-O_E-CARE_CN': 'CEG-O_E-CARE',
        'CEG-O_E-CARE_EN': 'CEG-O_E-CARE',
        # counterfactual_reasoning/
        'CR-B_det-counterfactual_CN': 'CLADDER',
        'CR-B_det-counterfactual_EN': 'CLADDER',
        'CR-C_CRASS_CN': 'CR-C_CRASS',
        'CR-C_CRASS_EN': 'CR-C_CRASS',
        # effect_of_the_treatment_on_the_treated/
        'ETT-B_ETT-natural_CN': 'Natural',
        'ETT-B_ETT-natural_EN': 'Natural',
        'ETT-P_ETT-basic_CN': 'Probability',
        'ETT-P_ETT-basic_EN': 'Probability',
        'ETT-P_ETT-hard_CN': 'Probability',
        'ETT-P_ETT-hard_EN': 'Probability',
        # natural_direct_effect/
        'NDE-B_NDE-natural_CN': 'Natural',
        'NDE-B_NDE-natural_EN': 'Natural',
        'NDE-P_NDE-basic_CN': 'Probability',
        'NDE-P_NDE-basic_EN': 'Probability',
        'NDE-P_NDE-hard_CN': 'Probability',
        'NDE-P_NDE-hard_EN': 'Probability',
        # natural_indirect_effect/
        'NIE-B_NIE-natural_CN': 'Natural',
        'NIE-B_NIE-natural_EN': 'Natural',
        'NIE-P_NIE-basic_CN': 'Probability',
        'NIE-P_NIE-basic_EN': 'Probability',
        'NIE-P_NIE-hard_CN': 'Probability',
        'NIE-P_NIE-hard_EN': 'Probability',
        # probability_of_necessity/
        'PN-P_PN-basic_CN': 'Probability',
        'PN-P_PN-basic_EN': 'Probability',
        'PN-P_PN-hard_CN': 'Probability',
        'PN-P_PN-hard_EN': 'Probability',
        # probability_of_sufficiency/
        'PS-P_PS-basic_CN': 'Probability',
        'PS-P_PS-basic_EN': 'Probability',
        'PS-P_PS-hard_CN': 'Probability',
        'PS-P_PS-hard_EN': 'Probability',
        # intervention/
        # average_treatment_effect/
        'ATE-B_ATE-natural_CN': 'Natural',
        'ATE-B_ATE-natural_EN': 'Natural',
        'ATE-P_ATE-basic_CN': 'Probability',
        'ATE-P_ATE-basic_EN': 'Probability',
        'ATE-P_ATE-hard_CN': 'Probability',
        'ATE-P_ATE-hard_EN': 'Probability',
        # backdoor_adjustment_set/
        'BAS-B_backadj_CN': 'CLADDER',
        'BAS-B_backadj_EN': 'CLADDER',
        'BAS-C_max-BAS_CN': 'AS',
        'BAS-C_max-BAS_EN': 'AS',
        'BAS-C_min-BAS_CN': 'AS',
        'BAS-C_min-BAS_EN': 'AS',
        'BAS-C_mix-BAS_CN': 'AS',
        'BAS-C_mix-BAS_EN': 'AS',
        # causal_effect_identification/
        'CEI-B_0.2-UC_CN': 'CEI-B',
        'CEI-B_0.2-UC_EN': 'CEI-B',
        'CEI-B_0.4-UC_CN': 'CEI-B',
        'CEI-B_0.4-UC_EN': 'CEI-B',
        'CEI-B_0.6-UC_CN': 'CEI-B',
        'CEI-B_0.6-UC_EN': 'CEI-B',
        'CEI-B_0.8-UC_CN': 'CEI-B',
        'CEI-B_0.8-UC_EN': 'CEI-B',
        # collider_bias/
        'CB-B_collider-bias_CN': 'CLADDER',
        'CB-B_collider-bias_EN': 'CLADDER',
        # controlled_direct_effect/
        'CDE-B_CDE-natural_CN': 'Natural',
        'CDE-B_CDE-natural_EN': 'Natural',
        'CDE-P_CDE-basic_CN': 'Probability',
        'CDE-P_CDE-basic_EN': 'Probability',
        'CDE-P_CDE-hard_CN': 'Probability',
        'CDE-P_CDE-hard_EN': 'Probability',
        # frontdoor_adjustment_set/
        'FAS-C_FAS_CN': 'AS',
        'FAS-C_FAS_EN': 'AS',
        # instrumental_variable/
        'IV-C_CaLM-IV_CN': 'AS',
        'IV-C_CaLM-IV_EN': 'AS',
    }

    labeling_module_name = task_to_labeling_module_map.get(task)
    if labeling_module_name:
        labeling_module = importlib.import_module(
            f'opencompass.datasets.calm.evaluation.labeling.{labeling_module_name}'
        )
        get_ground_truth_label = labeling_module.get_gt_label
        get_predicted_label = labeling_module.get_pred_label
    else:
        raise NotImplementedError(
            f'No labeling functions found for task {task}.')

    accuracy_module_name = task_to_accuracy_module_map.get(task)
    if accuracy_module_name:
        accuracy_module = importlib.import_module(
            f'opencompass.datasets.calm.evaluation.accuracy.{accuracy_module_name}'
        )
        get_accuracy = accuracy_module.compute_acc
    else:
        raise NotImplementedError(
            f'No accuracy functions found for task {task}.')

    return get_ground_truth_label, get_predicted_label, get_accuracy


def compute_core_metrics(items, task, prompt_style, gt_items):
    """Computes core metrics for a given set of items based on the ground truth
    items.

    Args:
        items (list): The list of items generated by the model.
        task (str): The task type.
        prompt_style (str): The prompt style.
        gt_items (list): The list of ground truth items.

    Returns:
        tuple: A tuple containing the computed core metrics dictionary and the list of predicted labels.

    Raises:
        AssertionError: If there is an index mismatch between items and ground truth items.
    """
    core_metrics_dict = {}
    get_gt_label, get_pred_label, compute_acc = initialize_core_metric_evaluation_components(
        task)
    gt_list, pred_list, pred_AP_list = [], [], []

    # get labels
    assert len(items) == len(
        gt_items), 'Length mismatch between items and ground truth items.'
    for item, gt_item in zip(items, gt_items):
        gt_label = get_gt_label(gt_item)

        type = task.split('-')[0]
        pred_label = get_pred_label(item, gt_item, prompt_style, type)
        gt_list.append(gt_label)
        pred_list.append(pred_label)

    # compute metrics
    core_metrics_dict['Accuracy'] = compute_acc(gt_list, pred_list)

    return core_metrics_dict, pred_list
