from opencompass.summarizers.default import DefaultSummarizer
import functools
import math


def calculate_opi(scores):
    assert len(scores) == 16, 'Expected 16 scores for OPI group, got {}'.format(len(scores))
    sum_score = 0
    for i in range(16):
        sum_score += 100 * scores[i]
    return sum_score / 16


def calculate_smol(scores):
    assert len(scores) == 15, 'Expected 15 scores for SmolInstruct group, got {}'.format(len(scores))
    sum_score = sum(scores[:2] + scores[3:8] + scores[10:])
    sum_score += 100 * scores[2]  # meteor_score
    sum_score += max(0, 100 * (4 - scores[8]) / 4)
    sum_score += max(0, 100 * (2 - scores[9]) / 2)
    return sum_score / 15


def calculate_mol(scores):
    assert len(scores) == 17, 'Expected 17 scores for Mol_Instructions group, got {}'.format(len(scores))
    sum_score = sum(scores[3:4] + scores[5:6] + scores[12:])
    sum_score += 100 * sum(scores[:2] + scores[4:5] + scores[7:12])
    sum_score += max(0, 100 * (10 - scores[6]) / 10)
    return sum_score / 17


def calculate_llm4mat(scores):
    assert len(scores) == 65, 'Expected 65 scores for LLM4Mat group, got {}'.format(len(scores))
    sum_score = 100 * sum(scores[:5]) + sum(scores[5:])
    return sum_score / 65


def calculate_unconditional_gen(scores):
    assert len(scores) == 4, 'Expected 4 scores for Unconditional_Generation group, got {}'.format(len(scores))
    sum_score = scores[0] + min(100, (0 - scores[1]) / 10) + 100 * sum(scores[2:])
    return sum_score / 4


scireasoner_summary_groups = [
    {
        'name': 'Bio_Instruction',
        'subsets': [['SciReasoner-bio_instruction-antibody_antigen', 'MCC'], ['SciReasoner-bio_instruction-rna_protein_interaction', 'MCC'], ['SciReasoner-bio_instruction-emp', 'MCC'],
                    ['SciReasoner-bio_instruction-enhancer_activity', 'PCC'], ['SciReasoner-bio_instruction-tf_m', 'MCC'], ['SciReasoner-bio_instruction-Isoform', 'R2'],
                    ['SciReasoner-bio_instruction-Modification', 'AUC'], ['SciReasoner-bio_instruction-MeanRibosomeLoading', 'R2'],
                    ['SciReasoner-bio_instruction-ProgrammableRNASwitches', 'R2'], ['SciReasoner-bio_instruction-CRISPROnTarget', 'spearman'],
                    ['SciReasoner-bio_instruction-promoter_enhancer_interaction', 'MCC'], ['SciReasoner-bio_instruction-sirnaEfficiency', 'mixed_score'],
                    ['SciReasoner-bio_instruction-cpd', 'MCC'], ['SciReasoner-bio_instruction-pd', 'MCC'], ['SciReasoner-bio_instruction-tf_h', 'MCC']],
        'metric': 'naive_average'
    },
    {
        'name': 'GUE',
        'subsets': [['SciReasoner-Gue_cpd-prom_core_all', 'matthews_correlation_all'],
                    ['SciReasoner-Gue_cpd-prom_core_notata', 'matthews_correlation_all'],
                    ['SciReasoner-Gue_cpd-prom_core_tata', 'matthews_correlation_all'],
                    ['SciReasoner-Gue_pd-prom_300_all', 'matthews_correlation_all'],
                    ['SciReasoner-Gue_pd-prom_300_notata', 'matthews_correlation_all'],
                    ['SciReasoner-Gue_pd-prom_300_tata', 'matthews_correlation_all'],
                    ['SciReasoner-Gue_tf-h-0', 'matthews_correlation_all'], ['SciReasoner-Gue_tf-h-1', 'matthews_correlation_all'],
                    ['SciReasoner-Gue_tf-h-2', 'matthews_correlation_all'], ['SciReasoner-Gue_tf-h-3', 'matthews_correlation_all'],
                    ['SciReasoner-Gue_tf-h-4', 'matthews_correlation_all']],
        'metric': 'naive_average'
    },
    {
        'name': 'OPI',
        'subsets': [['SciReasoner-Opi_EC_number_CLEAN_EC_number_new', 'Accuracy'],
                    ['SciReasoner-Opi_EC_number_CLEAN_EC_number_price', 'Accuracy'], ['SciReasoner-Opi_Fold_type_fold_type', 'Accuracy'],
                    ['SciReasoner-Opi_Function_CASPSimilarSeq_function', 'ROUGE-L'],
                    ['SciReasoner-Opi_Function_IDFilterSeq_function', 'ROUGE-L'], ['SciReasoner-Opi_Function_UniProtSeq_function', 'ROUGE-L'],
                    ['SciReasoner-Opi_gName2Cancer_gene_name_to_cancer', 'F1 Score'],
                    ['SciReasoner-Opi_GO_CASPSimilarSeq_go', 'F1 Score'], ['SciReasoner-Opi_GO_IDFilterSeq_go', 'F1 Score'],
                    ['SciReasoner-Opi_GO_UniProtSeq_go', 'F1 Score'], ['SciReasoner-Opi_gSymbol2Cancer_gene_symbol_to_cancer', 'F1 Score'],
                    ['SciReasoner-Opi_gSymbol2Tissue_gene_symbol_to_tissue', 'F1 Score'],
                    ['SciReasoner-Opi_Keywords_CASPSimilarSeq_keywords', 'F1 Score'],
                    ['SciReasoner-Opi_Keywords_IDFilterSeq_keywords', 'F1 Score'],
                    ['SciReasoner-Opi_Keywords_UniProtSeq_keywords', 'F1 Score'],
                    ['SciReasoner-Opi_Subcellular_localization_subcell_loc', 'Accuracy']],
        'function': calculate_opi
    },
    {
        'name': 'PEER',
        'subsets': [['SciReasoner-PEER_solubility', 'accuracy'], ['SciReasoner-PEER_stability', 'accuracy'], ['SciReasoner-PEER_human_ppi', 'accuracy'],
                    ['SciReasoner-PEER_yeast_ppi', 'accuracy']],
        'metric': 'naive_average'
    },
    {
        'name': 'SmolInstruct',
        'subsets': [['SciReasoner-smol_forward_synthesis', 'top1_exact_match'],
                    ['SciReasoner-smol_retrosynthesis', 'top1_exact_match'], ['SciReasoner-smol_molecule_captioning', 'meteor_score'],
                    ['SciReasoner-smol_molecule_generation', 'top1_exact_match'],
                    ['SciReasoner-smol_name_conversion-i2f', 'top1_ele_match'],
                    ['SciReasoner-smol_name_conversion-i2s', 'top1_exact_match'],
                    ['SciReasoner-smol_name_conversion-s2f', 'top1_ele_match'],
                    ['SciReasoner-smol_name_conversion-s2i', 'top1_split_match'],
                    ['SciReasoner-smol_property_prediction-esol', 'RMSE'],
                    ['SciReasoner-smol_property_prediction-lipo', 'RMSE'], ['SciReasoner-smol_property_prediction-bbbp', 'accuracy'],
                    ['SciReasoner-smol_property_prediction-clintox', 'accuracy'],
                    ['SciReasoner-smol_property_prediction-hiv', 'accuracy'],
                    ['SciReasoner-smol_property_prediction-sider', 'accuracy'],
                    ['SciReasoner-retrosynthesis_USPTO_50K', 'Top-1 Accuracy']],
        'function': calculate_smol
    },
    {
        'name': 'Mol_Instructions',
        'subsets': [['SciReasoner-mol_instruction_chemical_disease_interaction_extraction', 'f1'],
                    ['SciReasoner-mol_instruction_chemical_entity_recognition', 'f1'],
                    ['SciReasoner-mol_instruction_chemical_protein_interaction_extraction', 'f1'],
                    ['SciReasoner-mol_instruction_multi_choice_question', 'accuracy'],
                    ['SciReasoner-mol_instruction_open_question', 'bert_score'],
                    ['SciReasoner-mol_instruction_true_or_false_question', 'accuracy'],
                    ['SciReasoner-mol_instruction_property_prediction_str', 'mae'],
                    ['SciReasoner-mol_instruction_description_guided_molecule_design', 'exact_match_score'],
                    ['SciReasoner-mol_instruction_forward_reaction_prediction', 'exact_match_score'],
                    ['SciReasoner-mol_instruction_retrosynthesis', 'exact_match_score'],
                    ['SciReasoner-mol_instruction_reagent_prediction', 'exact_match_score'],
                    ['SciReasoner-mol_instruction_molecular_description_generation', 'rougeL'],
                    ['SciReasoner-mol_instruction_catalytic_activity', 'rougeL'],
                    ['SciReasoner-mol_instruction_domain_motif', 'rougeL'],
                    ['SciReasoner-mol_instruction_general_function', 'rougeL'],
                    ['SciReasoner-mol_instruction_protein_function', 'rougeL'],
                    ['SciReasoner-mol_instruction_protein_design', 'Max SW score']],
        'function': calculate_mol
    },
    {
        'name': 'LLM4Mat',
        'subsets': [['SciReasoner-LLM4Mat_MP_IsStable', 'AUC'], ['SciReasoner-LLM4Mat_MP_IsGapDirect', 'AUC'], ['SciReasoner-LLM4Mat_SNUMAT_IsDirect', 'AUC'],
                    ['SciReasoner-LLM4Mat_SNUMAT_IsDirect_HSE', 'AUC'], ['SciReasoner-LLM4Mat_SNUMAT_SOC', 'AUC'], ['SciReasoner-LLM4Mat_MP_FEPA', 'MAD/MAE'],
                    ['SciReasoner-LLM4Mat_MP_Bandgap', 'MAD/MAE'], ['SciReasoner-LLM4Mat_MP_EPA', 'MAD/MAE'], ['SciReasoner-LLM4Mat_MP_Ehull', 'MAD/MAE'],
                    ['SciReasoner-LLM4Mat_MP_Efermi', 'MAD/MAE'], ['SciReasoner-LLM4Mat_MP_Density', 'MAD/MAE'], ['SciReasoner-LLM4Mat_MP_DensityAtomic', 'MAD/MAE'],
                    ['SciReasoner-LLM4Mat_MP_Volume', 'MAD/MAE'], ['SciReasoner-LLM4Mat_JARVISDFT_FEPA', 'MAD/MAE'],
                    ['SciReasoner-LLM4Mat_JARVISDFT_Bandgap_OPT', 'MAD/MAE'], ['SciReasoner-LLM4Mat_JARVISDFT_TotEn', 'MAD/MAE'],
                    ['SciReasoner-LLM4Mat_JARVISDFT_Ehull', 'MAD/MAE'], ['SciReasoner-LLM4Mat_JARVISDFT_Bandgap_MBJ', 'MAD/MAE'],
                    ['SciReasoner-LLM4Mat_JARVISDFT_Kv', 'MAD/MAE'], ['SciReasoner-LLM4Mat_JARVISDFT_Gv', 'MAD/MAE'],
                    ['SciReasoner-LLM4Mat_JARVISDFT_SLME', 'MAD/MAE'], ['SciReasoner-LLM4Mat_JARVISDFT_Spillage', 'MAD/MAE'],
                    ['SciReasoner-LLM4Mat_JARVISDFT_Epsx_OPT', 'MAD/MAE'], ['SciReasoner-LLM4Mat_JARVISDFT_Dielectric_DFPT', 'MAD/MAE'],
                    ['SciReasoner-LLM4Mat_JARVISDFT_Max_Piezo_dij', 'MAD/MAE'], ['SciReasoner-LLM4Mat_JARVISDFT_Max_Piezo_eij', 'MAD/MAE'],
                    ['SciReasoner-LLM4Mat_JARVISDFT_MaxEFG', 'MAD/MAE'], ['SciReasoner-LLM4Mat_JARVISDFT_ExfEn', 'MAD/MAE'],
                    ['SciReasoner-LLM4Mat_JARVISDFT_AvgMe', 'MAD/MAE'], ['SciReasoner-LLM4Mat_JARVISDFT_nSeebeck', 'MAD/MAE'],
                    ['SciReasoner-LLM4Mat_JARVISDFT_nPF', 'MAD/MAE'], ['SciReasoner-LLM4Mat_JARVISDFT_pSeebeck', 'MAD/MAE'],
                    ['SciReasoner-LLM4Mat_JARVISDFT_pPF', 'MAD/MAE'], ['SciReasoner-LLM4Mat_SNUMAT_Bandgap_GGA', 'MAD/MAE'],
                    ['SciReasoner-LLM4Mat_SNUMAT_Bandgap_HSE', 'MAD/MAE'], ['SciReasoner-LLM4Mat_SNUMAT_Bandgap_GGA_Optical', 'MAD/MAE'],
                    ['SciReasoner-LLM4Mat_SNUMAT_Bandgap_HSE_Optical', 'MAD/MAE'], ['SciReasoner-LLM4Mat_GNoME_FEPA', 'MAD/MAE'],
                    ['SciReasoner-LLM4Mat_GNoME_DEPA', 'MAD/MAE'], ['SciReasoner-LLM4Mat_GNoME_Bandgap', 'MAD/MAE'],
                    ['SciReasoner-LLM4Mat_GNoME_TotEn', 'MAD/MAE'], ['SciReasoner-LLM4Mat_GNoME_Volume', 'MAD/MAE'],
                    ['SciReasoner-LLM4Mat_GNoME_Density', 'MAD/MAE'], ['SciReasoner-LLM4Mat_hMOF_MaxCO2', 'MAD/MAE'], ['SciReasoner-LLM4Mat_hMOF_MinCO2', 'MAD/MAE'],
                    ['SciReasoner-LLM4Mat_hMOF_LCD', 'MAD/MAE'], ['SciReasoner-LLM4Mat_hMOF_PLD', 'MAD/MAE'], ['SciReasoner-LLM4Mat_hMOF_VoidFraction', 'MAD/MAE'],
                    ['SciReasoner-LLM4Mat_hMOF_SA_m2g', 'MAD/MAE'], ['SciReasoner-LLM4Mat_hMOF_SA_m2cm3', 'MAD/MAE'],
                    ['SciReasoner-LLM4Mat_Cantor_HEA_FEPA', 'MAD/MAE'], ['SciReasoner-LLM4Mat_Cantor_HEA_EPA', 'MAD/MAE'],
                    ['SciReasoner-LLM4Mat_Cantor_HEA_Ehull', 'MAD/MAE'], ['SciReasoner-LLM4Mat_Cantor_HEA_VPA', 'MAD/MAE'],
                    ['SciReasoner-LLM4Mat_QMOF_TotEn', 'MAD/MAE'], ['SciReasoner-LLM4Mat_QMOF_Bandgap', 'MAD/MAE'], ['SciReasoner-LLM4Mat_QMOF_LCD', 'MAD/MAE'],
                    ['SciReasoner-LLM4Mat_QMOF_PLD', 'MAD/MAE'], ['SciReasoner-LLM4Mat_JARVISQETB_EPA', 'MAD/MAE'],
                    ['SciReasoner-LLM4Mat_JARVISQETB_IndirBandgap', 'MAD/MAE'], ['SciReasoner-LLM4Mat_JARVISQETB_FEPA', 'MAD/MAE'],
                    ['SciReasoner-LLM4Mat_JARVISQETB_TotEn', 'MAD/MAE'], ['SciReasoner-LLM4Mat_OQMD_Bandgap', 'MAD/MAE'],
                    ['SciReasoner-LLM4Mat_OQMD_FEPA', 'MAD/MAE'], ['SciReasoner-LLM4Mat_OMDB_Bandgap', 'MAD/MAE']],
        'function': calculate_llm4mat
    },
    {
        'name': 'Conditional_Generation',
        'subsets': [['SciReasoner-composition_to_material_generation', 'smact_validity_ratio_in_all_%'],
                    ['SciReasoner-bulk_modulus_to_material_generation', 'smact_validity_ratio_in_all_%']],
        'metric': 'naive_average'
    },
    {
        'name': 'Unconditional_Generation',
        'subsets': [['SciReasoner-unconditional_material_generation', 'smact_validity_ratio_in_all'],
                    ['SciReasoner-unconditional_RNA_generation', 'average_mfe'],
                    ['SciReasoner-unconditional_protein_generation', 'valid_rate'],
                    ['SciReasoner-unconditional_molecule_generation', 'validity']],
        'function': calculate_unconditional_gen
    }
]


class SciReasonerSummarizer(DefaultSummarizer):
    def __init__(self, mini_set=False, show_details=False, *args, **kwargs):
        """
        mini_set: 如果测的是mini版本需要开True，默认False
        show_details: 是否需要展示最底层的分数，默认不展示
        """
        super().__init__(*args, **kwargs)
        self.summary_groups = scireasoner_summary_groups
        if mini_set:
            for subset in self.summary_groups:
                subset['name'] = f"{subset['name']}-mini"
                for sub in subset['subsets']:
                    sub[0] = f'{sub[0]}-mini'
        self.dataset_abbrs = [sg['name'] for sg in self.summary_groups]
        if show_details:
            self.dataset_abbrs += [sub for sg in self.summary_groups for sub in sg['subsets']]

    def _calculate_group_metrics(self, raw_results, parsed_results, dataset_metrics, dataset_eval_mode):
        """The function calculates the numerical results for each group based
        on the configuration in summary_groups, and updates the contents of
        each dictionary accordingly."""
        summary_groups = self.summary_groups
        for sg in summary_groups:
            for model_abbr in self.model_abbrs:
                available_metrics, missing_metrics = [], []
                for i in sg['subsets']:
                    if isinstance(i, (list, tuple)):
                        # 自动修复输出无效的子集分数，默认设为0
                        if i[0] in parsed_results[model_abbr] and i[1] in parsed_results[model_abbr][i[0]]:
                            available_metrics.append(i)
                        else:
                            self.logger.warning(
                                f'Missing metric {i[1]} for dataset {i[0]} in model {model_abbr}, setting default value 0.')
                            parsed_results[model_abbr][i[0]] = {i[1]: 0.0}
                            available_metrics.append(i)
                        if not isinstance(parsed_results[model_abbr][i[0]][i[1]], (int, float)):
                            self.logger.warning(
                                f'Non-numeric metric {i[1]} for dataset {i[0]} in model {model_abbr}, setting default value 0.')
                            parsed_results[model_abbr][i[0]][i[1]] = 0.0
                        if math.isinf(parsed_results[model_abbr][i[0]][i[1]]) or math.isnan(parsed_results[model_abbr][i[0]][i[1]]):
                            parsed_results[model_abbr][i[0]][i[1]] = 0.0
                    else:
                        if i in parsed_results[model_abbr]:
                            available_metrics.append(i)
                        else:
                            missing_metrics.append(i)

                if len(available_metrics) == 0:
                    continue
                if len(missing_metrics) != 0:
                    raw_results[model_abbr][sg['name']] = {'error': 'missing metrics: {}'.format(missing_metrics)}
                    continue

                if 'metric' in sg:
                    default_metric = sg['metric']
                    need_smart_metric = False
                else:
                    need_smart_metric = True
                    if sg.get('std', False):
                        default_metric = 'standard_deviation'
                    elif sg.get('sum', False):
                        default_metric = 'sum'
                    elif sg.get('weights', []):
                        default_metric = 'weighted_average'
                    elif sg.get('harmonic_mean', False):
                        default_metric = 'harmonic_mean'
                    elif sg.get('function', None):
                        default_metric = 'function_score'
                    else:
                        default_metric = 'naive_average'

                scores, eval_modes, group_metrics = {}, [], None
                if any(isinstance(dataset_abbr, (list, tuple)) for dataset_abbr in sg['subsets']) and \
                        any(isinstance(dataset_abbr, str) for dataset_abbr in sg['subsets']):
                    raise NotImplementedError('mixed dataset_abbr type is not supported')

                if all(isinstance(dataset_abbr, (list, tuple)) for dataset_abbr in sg['subsets']):
                    group_metrics = [default_metric]
                    for dataset_abbr, metric in sg['subsets']:
                        scores.setdefault(default_metric, {})[dataset_abbr + '@' + metric] = \
                            parsed_results[model_abbr][dataset_abbr][metric]
                        eval_modes.append(dataset_eval_mode.get(dataset_abbr, 'unknown'))
                else:
                    group_metrics = list(functools.reduce(lambda a, b: a & b,
                                                          [set(dataset_metrics[dataset_abbr]) for dataset_abbr in
                                                           sg['subsets']]))
                    group_metrics.append(default_metric)
                    for metric in group_metrics:
                        for dataset_abbr in sg['subsets']:
                            if metric == default_metric:
                                metric_default = dataset_metrics[dataset_abbr][0]
                                scores.setdefault(default_metric, {})[dataset_abbr + '@' + metric_default] = \
                                    parsed_results[model_abbr][dataset_abbr][metric_default]
                                eval_modes.append(dataset_eval_mode.get(dataset_abbr, 'unknown'))
                            else:
                                scores.setdefault(metric, {})[dataset_abbr + '@' + metric] = \
                                    parsed_results[model_abbr][dataset_abbr][metric]
                                eval_modes.append(dataset_eval_mode.get(sg['subsets'][0], 'unknown'))
                result = {}
                for metric in scores:
                    if default_metric == 'standard_deviation':
                        avg = sum(scores[metric].values()) / len(scores[metric])
                        variance = sum((scores[metric][k] - avg) ** 2 for k in scores[metric]) / len(scores[metric])
                        scores[metric] = result[metric] = math.sqrt(variance)
                    elif default_metric == 'harmonic_mean':
                        # Check for non-positive values that would cause issues in harmonic mean
                        if any(scores[metric][k] <= 0 for k in scores[metric]):
                            self.logger.warning(
                                f'Non-positive values found when calculating harmonic mean for {sg["name"]}')
                            # Handle non-positive values (either skip or use a small positive value)
                            numerator = len(scores[metric])
                            denominator = sum(1 / max(scores[metric][k], 1) for k in scores[metric])
                        else:
                            numerator = len(scores[metric])
                            denominator = sum(1 / scores[metric][k] for k in scores[metric])
                        scores[metric] = result[metric] = numerator / denominator
                    elif default_metric == 'function_score':
                        func = sg['function']
                        score_list = [scores[metric][f'{sub[0]}@{sub[1]}'] for sub in sg['subsets']]
                        scores[metric] = result[metric] = func(score_list)
                    else:
                        if sg.get('weights', []):
                            # check sg['weights'][k] != 0 in case of scores[metric][k] is NaN
                            try:
                                numerator = sum(scores[metric][k] * sg['weights'][k] for k in sg['weights'] if
                                                sg['weights'][k] != 0)
                            except KeyError:
                                tmp_scores = {metric: {k.split('@')[0]: v for k, v in scores[metric].items()}}
                                numerator = sum(tmp_scores[metric][k] * sg['weights'][k] for k in sg['weights'] if
                                                sg['weights'][k] != 0)
                            denominator = sum(sg['weights'].values())
                        else:
                            numerator = sum(scores[metric].values())
                            denominator = len(scores[metric])
                        if default_metric == 'sum':
                            scores[metric] = result[metric] = numerator
                        else:
                            scores[metric] = result[metric] = numerator / denominator
                    eval_modes = list(set(eval_modes))
                    eval_mode = eval_modes[0] if len(eval_modes) == 1 else 'mixed'

                # add to global results
                raw_results[model_abbr].setdefault(sg['name'], {}).update(scores)
                parsed_results[model_abbr].setdefault(sg['name'], {}).update(result)
                dataset_metrics.setdefault(sg['name'], []).extend(group_metrics)
                dataset_eval_mode[sg['name']] = eval_mode
        breakpoint()
        return raw_results, parsed_results, dataset_metrics, dataset_eval_mode
