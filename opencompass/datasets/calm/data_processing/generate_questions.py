# flake8: noqa: E501
import importlib
from pathlib import Path

from ..utils.load_items import load_query_instances


def get_get_prompt_func(task):
    """Returns the appropriate prompt generation function based on the given
    task.

    Args:
        task (str): The name of the task for which the prompt function is required.

    Returns:
        function: The prompt generation function for the specified task.

    Raises:
        NotImplementedError: If no prompt function is found for the given task.
    """
    task_to_module_map = {
        # association/
        # correlation/
        'CORR-B_correlation_CN': 'CORR-B_correlation',
        'CORR-B_correlation_EN': 'CORR-B_correlation',
        # explaining_away_effect/
        'EAE-B_exp-away_CN': 'EAE-B_exp-away',
        'EAE-B_exp-away_EN': 'EAE-B_exp-away',
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
        'ECI-B_CTB_CN': 'ECI-B_CTB',
        'ECI-B_CTB_EN': 'ECI-B_CTB',
        'ECI-B_ESC_CN': 'ECI-B_ESC',
        'ECI-B_ESC_EN': 'ECI-B_ESC',
        'ECI-B_MAVEN-ERE_CN': 'ECI-B_MAVEN-ERE',
        'ECI-B_MAVEN-ERE_EN': 'ECI-B_MAVEN-ERE',
        # pairwise_causal_discovery/
        'PCD-B_COPA_CN': 'PCD-B_COPA',
        'PCD-B_COPA_EN': 'PCD-B_COPA',
        'PCD-B_E-CARE_CN': 'PCD-B_E-CARE',
        'PCD-B_E-CARE_EN': 'PCD-B_E-CARE',
        'PCD-C_COPA_CN': 'PCD-C_COPA',
        'PCD-C_COPA_EN': 'PCD-C_COPA',
        'PCD-C_E-CARE_CN': 'PCD-C_E-CARE',
        'PCD-C_E-CARE_EN': 'PCD-C_E-CARE',
        # counterfactual/
        # actual_causality/
        'AC-B_causal_judgement_CN': 'AC-B_causal_judgement',
        'AC-B_causal_judgement_EN': 'AC-B_causal_judgement',
        # causal_explanation_generation/
        'CEG-O_E-CARE_CN': 'CEG-O_E-CARE',
        'CEG-O_E-CARE_EN': 'CEG-O_E-CARE',
        # counterfactual_reasoning/
        'CR-B_det-counterfactual_CN': 'CR-B_det-counterfactual',
        'CR-B_det-counterfactual_EN': 'CR-B_det-counterfactual',
        'CR-C_CRASS_CN': 'CR-C_CRASS',
        'CR-C_CRASS_EN': 'CR-C_CRASS',
        # effect_of_the_treatment_on_the_treated/
        'ETT-B_ETT-natural_CN': 'ETT',
        'ETT-B_ETT-natural_EN': 'ETT',
        'ETT-P_ETT-basic_CN': 'ETT',
        'ETT-P_ETT-basic_EN': 'ETT',
        'ETT-P_ETT-hard_CN': 'ETT',
        'ETT-P_ETT-hard_EN': 'ETT',
        # natural_direct_effect/
        'NDE-B_NDE-natural_CN': 'NDE',
        'NDE-B_NDE-natural_EN': 'NDE',
        'NDE-P_NDE-basic_CN': 'NDE',
        'NDE-P_NDE-basic_EN': 'NDE',
        'NDE-P_NDE-hard_CN': 'NDE',
        'NDE-P_NDE-hard_EN': 'NDE',
        # natural_indirect_effect/
        'NIE-B_NIE-natural_CN': 'NIE',
        'NIE-B_NIE-natural_EN': 'NIE',
        'NIE-P_NIE-basic_CN': 'NIE',
        'NIE-P_NIE-basic_EN': 'NIE',
        'NIE-P_NIE-hard_CN': 'NIE',
        'NIE-P_NIE-hard_EN': 'NIE',
        # probability_of_necessity/
        'PN-P_PN-basic_CN': 'PN',
        'PN-P_PN-basic_EN': 'PN',
        'PN-P_PN-hard_CN': 'PN',
        'PN-P_PN-hard_EN': 'PN',
        # probability_of_sufficiency/
        'PS-P_PS-basic_CN': 'PS',
        'PS-P_PS-basic_EN': 'PS',
        'PS-P_PS-hard_CN': 'PS',
        'PS-P_PS-hard_EN': 'PS',
        # intervention/
        # average_treatment_effect/
        'ATE-B_ATE-natural_CN': 'ATE',
        'ATE-B_ATE-natural_EN': 'ATE',
        'ATE-P_ATE-basic_CN': 'ATE',
        'ATE-P_ATE-basic_EN': 'ATE',
        'ATE-P_ATE-hard_CN': 'ATE',
        'ATE-P_ATE-hard_EN': 'ATE',
        # backdoor_adjustment_set/
        'BAS-B_backadj_CN': 'BAS-B_backadj',
        'BAS-B_backadj_EN': 'BAS-B_backadj',
        'BAS-C_max-BAS_CN': 'BAS-C_max-BAS',
        'BAS-C_max-BAS_EN': 'BAS-C_max-BAS',
        'BAS-C_min-BAS_CN': 'BAS-C_min-BAS',
        'BAS-C_min-BAS_EN': 'BAS-C_min-BAS',
        'BAS-C_mix-BAS_CN': 'BAS-C_mix-BAS',
        'BAS-C_mix-BAS_EN': 'BAS-C_mix-BAS',
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
        'CB-B_collider-bias_CN': 'CB-B_collider-bias',
        'CB-B_collider-bias_EN': 'CB-B_collider-bias',
        # controlled_direct_effect/
        'CDE-B_CDE-natural_CN': 'CDE',
        'CDE-B_CDE-natural_EN': 'CDE',
        'CDE-P_CDE-basic_CN': 'CDE',
        'CDE-P_CDE-basic_EN': 'CDE',
        'CDE-P_CDE-hard_CN': 'CDE',
        'CDE-P_CDE-hard_EN': 'CDE',
        # frontdoor_adjustment_set/
        'FAS-C_FAS_CN': 'FAS-C_FAS',
        'FAS-C_FAS_EN': 'FAS-C_FAS',
        # instrumental_variable/
        'IV-C_CaLM-IV_CN': 'IV-C_CaLM-IV',
        'IV-C_CaLM-IV_EN': 'IV-C_CaLM-IV',
    }

    module_name = task_to_module_map.get(task)

    if module_name:
        module = importlib.import_module(
            'opencompass.datasets.calm.data_processing.prompt.' + module_name)
        return module.get_prompt
    else:
        raise NotImplementedError(
            f'No get_prompt function found for task {task}.')


def generate_question_list(dataset_path, prompt_style):
    """Generates a list of questions from the dataset based on the specified
    prompt style.

    Args:
        dataset_path (str): The path to the dataset JSON file.
        prompt_style (str): The style of prompt to be used for generating questions.

    Returns:
        list: A list of question dictionaries, each containing an item from the dataset along with its corresponding question.

    Raises:
        AssertionError: If the task name and prompt style do not match the expected language suffix.
    """
    # Extract task name from dataset path
    dataset_path = Path(dataset_path)
    task_name = dataset_path.name[:-len('.json')]

    # Validate prompt style based on task language
    if task_name.endswith('CN'):
        assert prompt_style.endswith('-CN')
    else:
        assert not prompt_style.endswith('-CN')

    # Get prompt generation function based on task
    get_prompt_func = get_get_prompt_func(task=task_name)

    # Load items from dataset
    item_list = load_query_instances(dataset_path)
    question_list = []

    # Generate questions for each item in the dataset
    for idx, item in enumerate(item_list):
        question = get_prompt_func(task_name=task_name,
                                   prompt_style=prompt_style,
                                   item=item)
        question_list.append({
            'question': question,
            'gt_item': item,
        })
    return question_list
