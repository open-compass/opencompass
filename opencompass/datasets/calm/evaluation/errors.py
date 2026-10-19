# flake8: noqa: E501
import importlib
import json
import os
from pathlib import Path

from ..evaluation.core_metrics import \
    initialize_core_metric_evaluation_components


def initialize_error_identification_components(task, prompt_style):
    """Initialize error identification components.

    Args:
        task (str): The task for which error identification components are being initialized.
        prompt_style (str): The style of prompt for error identification.

    Returns:
        Module: The error identification module corresponding to the provided task and prompt style.
    """
    prompt_style_to_error_module_map = {
        'basic': 'basic_adversarial',
        'basic-CN': 'basic_adversarial',
        'adversarial-ignore': 'basic_adversarial',
        'adversarial-ignore-CN': 'basic_adversarial',
        'adversarial-doubt': 'basic_adversarial',
        'adversarial-doubt-CN': 'basic_adversarial',
        'zero-shot-IcL': 'icl',
        'zero-shot-IcL-CN': 'icl',
        'one-shot-IcL': 'icl',
        'one-shot-IcL-CN': 'icl',
        'three-shot-IcL': 'icl',
        'three-shot-IcL-CN': 'icl',
        'zero-shot-CoT': 'cot',
        'zero-shot-CoT-CN': 'cot',
        'manual-CoT': 'cot',
        'manual-CoT-CN': 'cot'
    }
    task_to_error_module_map = {
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
        'CA-B_FA_CN': 'CA-B',
        'CA-B_FA_EN': 'CA-B',
        'CA-B_FP_CN': 'CA-B',
        'CA-B_FP_EN': 'CA-B',
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

    error_task_module_name = task_to_error_module_map.get(task)
    error_prompt_module_name = prompt_style_to_error_module_map.get(
        prompt_style)

    if error_task_module_name and error_prompt_module_name:
        error_module = importlib.import_module(
            f'opencompass.datasets.calm.evaluation.error.{error_prompt_module_name}.{error_task_module_name}'
        )
        return error_module
    else:
        raise NotImplementedError(
            f'No get_score function found for task {task} and prompt {prompt_style}.'
        )


def identify_model_errors(items, task, prompt_style, gt_items):
    """Identify errors in model responses based on provided items, task, and
    prompt style.

    Args:
        items (list): A list of items containing model responses.
        task (str): The task type, note that CEG-O_E-CARE is not supported for error analysis.
        prompt_style (str): The style of prompt used, note that explicit-function is not supported for error analysis.
        gt_items (list): A list of ground truth items.

    Returns:
        dict: A dictionary containing error metrics for the model responses. (Same response to all questions, language inconsistency, limitation of instruction-following, repetition, empty response.)
    """
    if task == 'CEG-O_E-CARE' or prompt_style in [
            'explicit-function', 'explicit-function-CN'
    ]:
        print(
            'CEG-O_E-CARE and explicit-function prompts are not supported for error identification.'
        )
        return

    language_error, nonstandrad, repetition, empty = 0., 0., 0., 0.
    error_module = initialize_error_identification_components(
        task, prompt_style)
    get_gt_label, get_pred_label, compute_acc = initialize_core_metric_evaluation_components(
        task)
    pred_list = []

    for item, gt_item in zip(items, gt_items):
        pred_label = get_pred_label(item, gt_item, prompt_style,
                                    task.split('-')[0])
        pred_error = get_item_error(item, task, error_module, prompt_style)

        pred_list.append(pred_label)
        language_error += pred_error['language_error']
        nonstandrad += pred_error['nonstandrad']
        repetition += pred_error['repetition']
        empty += pred_error['empty']

    abnormalities = error_module.check_abnormality(pred_list)

    return {
        'Same response to all questions': 1 if abnormalities != 0 else 0,
        'Language inconsistency': language_error / len(pred_list),
        'Limitation of instruction-following': nonstandrad / len(pred_list),
        'Repetition': repetition / len(pred_list),
        'Empty response': empty / len(pred_list),
    }


def get_item_error(model_response, task, error_module, prompt_style):
    """Analyze errors in a single model response for a given task and prompt
    style.

    Args:
        model_response (str): The model's response to analyze.
        task (str): The task type.
        error_module: The error module containing error identification methods.
        prompt_style (str): The style of prompt used.

    Returns:
        dict: A dictionary containing error metrics for the model response. (Language inconsistency, nonstandardization, repetition, empty response.)
    """
    model_response = model_response.strip().lower()
    if 'CN' in task:
        language_error = error_module.contains_english(model_response)
    elif 'CN' not in task:
        language_error = error_module.contains_chinese(model_response)

    nonstandrad = error_module.check_standalization(model_response,
                                                    prompt_style,
                                                    type=task.split('-')[0])

    repetition = error_module.check_repetition(model_response)

    empty = error_module.check_empty(model_response)

    return {
        'language_error': language_error,
        'nonstandrad': nonstandrad,
        'repetition': repetition,
        'empty': empty,
    }
