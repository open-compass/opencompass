task_hiearchy_dict = {
    # association/
        # correlation/
            'CORR-B_correlation_CN':'association/correlation/',
            'CORR-B_correlation_EN':'association/correlation/',
        # explaining_away_effect/
            'EAE-B_exp-away_CN':'association/explaining_away_effect/',
            'EAE-B_exp-away_EN':'association/explaining_away_effect/',
    # causal_discovery/
        # abstract_reasoning/
            'AR-B_CaLM-AR_CN':'causal_discovery/abstract_reasoning/',
            'AR-B_CaLM-AR_EN':'causal_discovery/abstract_reasoning/',
        # causal_attribution/
            'CA-B_FA_CN':'causal_discovery/causal_attribution/',
            'CA-B_FA_EN':'causal_discovery/causal_attribution/',
            'CA-B_FP_CN':'causal_discovery/causal_attribution/',
            'CA-B_FP_EN':'causal_discovery/causal_attribution/',
        # event_causality_identification/
            'ECI-B_CTB_CN':'causal_discovery/event_causality_identification/',
            'ECI-B_CTB_EN':'causal_discovery/event_causality_identification/',
            'ECI-B_ESC_CN':'causal_discovery/event_causality_identification/',
            'ECI-B_ESC_EN':'causal_discovery/event_causality_identification/',
            'ECI-B_MAVEN-ERE_CN':'causal_discovery/event_causality_identification/',
            'ECI-B_MAVEN-ERE_EN':'causal_discovery/event_causality_identification/',
        # pairwise_causal_discovery/
            'PCD-B_COPA_CN':'causal_discovery/pairwise_causal_discovery/',
            'PCD-B_COPA_EN':'causal_discovery/pairwise_causal_discovery/',
            'PCD-B_E-CARE_CN':'causal_discovery/pairwise_causal_discovery/',
            'PCD-B_E-CARE_EN':'causal_discovery/pairwise_causal_discovery/',
            'PCD-C_COPA_CN':'causal_discovery/pairwise_causal_discovery/',
            'PCD-C_COPA_EN':'causal_discovery/pairwise_causal_discovery/',
            'PCD-C_E-CARE_CN':'causal_discovery/pairwise_causal_discovery/',
            'PCD-C_E-CARE_EN':'causal_discovery/pairwise_causal_discovery/',
    # counterfactual/
        # actual_causality/
            'AC-B_causal_judgement_CN':'counterfactual/actual_causality/',
            'AC-B_causal_judgement_EN':'counterfactual/actual_causality/',
        # causal_explanation_generation/
            'CEG-O_E-CARE_CN':'counterfactual/causal_explanation_generation/',
            'CEG-O_E-CARE_EN':'counterfactual/causal_explanation_generation/',
        # counterfactual_reasoning/
            'CR-B_det-counterfactual_CN':'counterfactual/counterfactual_reasoning/',
            'CR-B_det-counterfactual_EN':'counterfactual/counterfactual_reasoning/',
            'CR-C_CRASS_CN':'counterfactual/counterfactual_reasoning/',
            'CR-C_CRASS_EN':'counterfactual/counterfactual_reasoning/',
        # effect_of_the_treatment_on_the_treated/
            'ETT-B_ETT-natural_CN':'counterfactual/effect_of_the_treatment_on_the_treated/',
            'ETT-B_ETT-natural_EN':'counterfactual/effect_of_the_treatment_on_the_treated/',
            'ETT-P_ETT-basic_CN':'counterfactual/effect_of_the_treatment_on_the_treated/',
            'ETT-P_ETT-basic_EN':'counterfactual/effect_of_the_treatment_on_the_treated/',
            'ETT-P_ETT-hard_CN':'counterfactual/effect_of_the_treatment_on_the_treated/',
            'ETT-P_ETT-hard_EN':'counterfactual/effect_of_the_treatment_on_the_treated/',
        # natural_direct_effect/
            'NDE-B_NDE-natural_CN':'counterfactual/natural_direct_effect/',
            'NDE-B_NDE-natural_EN':'counterfactual/natural_direct_effect/',
            'NDE-P_NDE-basic_CN':'counterfactual/natural_direct_effect/',
            'NDE-P_NDE-basic_EN':'counterfactual/natural_direct_effect/',
            'NDE-P_NDE-hard_CN':'counterfactual/natural_direct_effect/',
            'NDE-P_NDE-hard_EN':'counterfactual/natural_direct_effect/',
        # natural_indirect_effect/
            'NIE-B_NIE-natural_CN':'counterfactual/natural_indirect_effect/',
            'NIE-B_NIE-natural_EN':'counterfactual/natural_indirect_effect/',
            'NIE-P_NIE-basic_CN':'counterfactual/natural_indirect_effect/',
            'NIE-P_NIE-basic_EN':'counterfactual/natural_indirect_effect/',
            'NIE-P_NIE-hard_CN':'counterfactual/natural_indirect_effect/',
            'NIE-P_NIE-hard_EN':'counterfactual/natural_indirect_effect/',
        # probability_of_necessity/
            'PN-P_PN-basic_CN':'counterfactual/probability_of_necessity/',
            'PN-P_PN-basic_EN':'counterfactual/probability_of_necessity/',
            'PN-P_PN-hard_CN':'counterfactual/probability_of_necessity/',
            'PN-P_PN-hard_EN':'counterfactual/probability_of_necessity/',
        # probability_of_sufficiency/
            'PS-P_PS-basic_CN':'counterfactual/probability_of_sufficiency/',
            'PS-P_PS-basic_EN':'counterfactual/probability_of_sufficiency/',
            'PS-P_PS-hard_CN':'counterfactual/probability_of_sufficiency/',
            'PS-P_PS-hard_EN':'counterfactual/probability_of_sufficiency/',
    # intervention/
        # average_treatment_effect/
            'ATE-B_ATE-natural_CN':'intervention/average_treatment_effect/',
            'ATE-B_ATE-natural_EN':'intervention/average_treatment_effect/',
            'ATE-P_ATE-basic_CN':'intervention/average_treatment_effect/',
            'ATE-P_ATE-basic_EN':'intervention/average_treatment_effect/',
            'ATE-P_ATE-hard_CN':'intervention/average_treatment_effect/',
            'ATE-P_ATE-hard_EN':'intervention/average_treatment_effect/',
        # backdoor_adjustment_set/
            'BAS-B_backadj_CN':'intervention/backdoor_adjustment_set/',
            'BAS-B_backadj_EN':'intervention/backdoor_adjustment_set/',
            'BAS-C_max-BAS_CN':'intervention/backdoor_adjustment_set/',
            'BAS-C_max-BAS_EN':'intervention/backdoor_adjustment_set/',
            'BAS-C_min-BAS_CN':'intervention/backdoor_adjustment_set/',
            'BAS-C_min-BAS_EN':'intervention/backdoor_adjustment_set/',
            'BAS-C_mix-BAS_CN':'intervention/backdoor_adjustment_set/',
            'BAS-C_mix-BAS_EN':'intervention/backdoor_adjustment_set/',
        # causal_effect_identification/
            'CEI-B_0.2-UC_CN':'intervention/causal_effect_identification/',
            'CEI-B_0.2-UC_EN':'intervention/causal_effect_identification/',
            'CEI-B_0.4-UC_CN':'intervention/causal_effect_identification/',
            'CEI-B_0.4-UC_EN':'intervention/causal_effect_identification/',
            'CEI-B_0.6-UC_CN':'intervention/causal_effect_identification/',
            'CEI-B_0.6-UC_EN':'intervention/causal_effect_identification/',
            'CEI-B_0.8-UC_CN':'intervention/causal_effect_identification/',
            'CEI-B_0.8-UC_EN':'intervention/causal_effect_identification/',
        # collider_bias/
            'CB-B_collider-bias_CN':'intervention/collider_bias/',
            'CB-B_collider-bias_EN':'intervention/collider_bias/',
        # controlled_direct_effect/
            'CDE-B_CDE-natural_CN':'intervention/controlled_direct_effect/',
            'CDE-B_CDE-natural_EN':'intervention/controlled_direct_effect/',
            'CDE-P_CDE-basic_CN':'intervention/controlled_direct_effect/',
            'CDE-P_CDE-basic_EN':'intervention/controlled_direct_effect/',
            'CDE-P_CDE-hard_CN':'intervention/controlled_direct_effect/',
            'CDE-P_CDE-hard_EN':'intervention/controlled_direct_effect/',
        # frontdoor_adjustment_set/
            'FAS-C_FAS_CN':'intervention/frontdoor_adjustment_set/',
            'FAS-C_FAS_EN':'intervention/frontdoor_adjustment_set/',
        # instrumental_variable/
            'IV-C_CaLM-IV_CN':'intervention/instrumental_variable/',
            'IV-C_CaLM-IV_EN':'intervention/instrumental_variable/',}
dict_keys = list(task_hiearchy_dict.keys())
error_dict = {'Same response to all questions':[],
               'Language inconsistency':[],
               'Limitation of instruction-following':[],
               'Repetition':[],
               'Empty response':[],}

for error in error_dict:
    for key in dict_keys:
        if 'CEG-O_E-CARE' in key:
            continue
        error_dict[error].append([f'calm_{key}', error])

English_avg = []
Chinese_avg = []
for key in dict_keys:
    if key.endswith('EN'):
        English_avg.append([f'calm_{key}', 'Accuracy'])
    else:
        assert key.endswith('CN')
        Chinese_avg.append([f'calm_{key}', 'Accuracy'])

calm_summary_groups = [
    # English Average
    {'name': 'English Average', 'subsets': English_avg},

    # Chinese Average
    {'name': 'Chinese Average', 'subsets': Chinese_avg},

    # Accuracy Average
    {'name': 'Accuracy Average', 'subsets': ['English Average', 'Chinese Average']},
]
for error in error_dict:
    calm_summary_groups.append({'name': error+' Average', 'subsets': error_dict[error]})

summarizer = dict(
    dataset_abbrs = [
        '###### CALM-Lite Accuracy ######',
        'Accuracy Average',
        'English Average',
        'Chinese Average',

        '###### CALM-Lite Errors ######',
        'Same response to all questions Average',
        'Language inconsistency Average',
        'Limitation of instruction-following Average',
        'Repetition Average',
        'Empty response Average',
    ],
    summary_groups=calm_summary_groups,
)
