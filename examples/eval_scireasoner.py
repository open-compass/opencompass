from mmengine.config import read_base

with read_base():
    # scireasoner
    from opencompass.configs.datasets.bio_instruction.bio_instruction import bio_instruction_datasets, \
        mini_bio_instruction_datasets
    from opencompass.configs.datasets.composition_material.composition_material_gen import \
        composition_material_datasets, mini_composition_material_datasets
    from opencompass.configs.datasets.GUE.GUE_gen import GUE_datasets, mini_GUE_datasets
    from opencompass.configs.datasets.LLM4Chem.all_datasets import all_datasets as smol_datasets, \
        mini_all_datasets as mini_smol_datasets
    from opencompass.configs.datasets.LLM4Chem.retrosynthesis import \
        Retrosynthesis_datasets as Retrosynthesis_uspto50k_datasets, \
        mini_Retrosynthesis_datasets as mini_Retrosynthesis_uspto50k_datasets
    from opencompass.configs.datasets.LLM4Mat.LLM4Mat_gen import LLM4Mat_datasets, mini_LLM4Mat_datasets
    from opencompass.configs.datasets.modulus_material.bulk_modulus_material_gen import modulus_material_datasets, \
        mini_modulus_material_datasets
    from opencompass.configs.datasets.Mol_Instructions.biotext import mol_biotext_datasets, mini_mol_biotext_datasets
    from opencompass.configs.datasets.Mol_Instructions.molecule import mol_mol_datasets, mini_mol_mol_datasets
    from opencompass.configs.datasets.Mol_Instructions.protein import mol_protein_datasets, mini_mol_protein_datasets
    from opencompass.configs.datasets.opi.all_datasets import all_datasets as opi_datasets, \
        mini_all_datasets as mini_opi_datasets
    from opencompass.configs.datasets.PEER.peer import PEER_datasets, mini_PEER_datasets
    from opencompass.configs.datasets.uncond_material.unconditional_material_gen import uncond_material_datasets, \
        mini_uncond_material_datasets
    from opencompass.configs.datasets.uncond_RNA.unconditional_RNA_gen import uncond_RNA_datasets, \
        mini_uncond_RNA_datasets
    from opencompass.configs.datasets.unconditional_protein_generation.UPG import \
        UPG_datasets as uncond_protein_datasets, mini_UPG_datasets as mini_uncond_protein_datasets
    from opencompass.configs.datasets.unconditional_molecule_generation.UMG import UMG_Datasets, mini_UMG_Datasets

# # full eval set
# datasets = bio_instruction_datasets + composition_material_datasets + GUE_datasets + smol_datasets + \
#            Retrosynthesis_uspto50k_datasets + LLM4Mat_datasets + modulus_material_datasets + \
#            mol_biotext_datasets + mol_mol_datasets + mol_protein_datasets + opi_datasets + PEER_datasets + \
#            uncond_material_datasets + uncond_RNA_datasets + uncond_protein_datasets + UMG_Datasets

# mini eval set
datasets = mini_bio_instruction_datasets + mini_composition_material_datasets + mini_GUE_datasets + mini_smol_datasets + \
           mini_Retrosynthesis_uspto50k_datasets + mini_LLM4Mat_datasets + mini_modulus_material_datasets + \
           mini_mol_biotext_datasets + mini_mol_mol_datasets + mini_mol_protein_datasets + mini_opi_datasets + mini_PEER_datasets + \
           mini_uncond_material_datasets + mini_uncond_RNA_datasets + mini_uncond_protein_datasets + mini_UMG_Datasets

# # full set summarizer
# summarizer = dict(
#     dataset_abbrs=[
#         ['antibody_antigen', 'MCC'], ['rna_protein_interaction', 'MCC'], ['emp', 'MCC'],
#         ['enhancer_activity', 'PCC'], ['tf_m', 'MCC'], ['Isoform', 'R2'], ['Modification', 'AUC'],
#         ['MeanRibosomeLoading', 'R2'], ['ProgrammableRNASwitches', 'R2'], ['CRISPROnTarget', 'spearman'],
#         ['promoter_enhancer_interaction', 'MCC'], ['sirnaEfficiency', 'mixed_score'], ['cpd', 'MCC'],
#         ['pd', 'MCC'], ['tf_h', 'MCC'],
#         ['cpd-prom_core_all', 'matthews_correlation_all'],
#         ['cpd-prom_core_notata', 'matthews_correlation_all'],
#         ['cpd-prom_core_tata', 'matthews_correlation_all'], ['pd-prom_300_all', 'matthews_correlation_all'],
#         ['pd-prom_300_notata', 'matthews_correlation_all'], ['pd-prom_300_tata', 'matthews_correlation_all'],
#         ['tf-h-0', 'matthews_correlation_all'], ['tf-h-1', 'matthews_correlation_all'],
#         ['tf-h-2', 'matthews_correlation_all'], ['tf-h-3', 'matthews_correlation_all'],
#         ['tf-h-4', 'matthews_correlation_all'], ['smol_forward_synthesis', 'top1_exact_match'],
#         ['smol_retrosynthesis', 'top1_exact_match'], ['smol_molecule_captioning', 'meteor_score'],
#         ['smol_molecule_generation', 'top1_exact_match'], ['smol_name_conversion-i2f', 'top1_ele_match'],
#         ['smol_name_conversion-i2s', 'top1_exact_match'], ['smol_name_conversion-s2f', 'top1_ele_match'],
#         ['smol_name_conversion-s2i', 'top1_split_match'], ['smol_property_prediction-esol', 'RMSE'],
#         ['smol_property_prediction-lipo', 'RMSE'], ['smol_property_prediction-bbbp', 'accuracy'],
#         ['smol_property_prediction-clintox', 'accuracy'], ['smol_property_prediction-hiv', 'accuracy'],
#         ['smol_property_prediction-sider', 'accuracy'], ['retrosynthesis_USPTO_50K', 'Top-1 Accuracy'],
#         ['MP_IsStable', 'AUC'], ['MP_IsGapDirect', 'AUC'], ['SNUMAT_IsDirect', 'AUC'],
#         ['SNUMAT_IsDirect_HSE', 'AUC'], ['SNUMAT_SOC', 'AUC'], ['MP_FEPA', 'MAD/MAE'],
#         ['MP_Bandgap', 'MAD/MAE'], ['MP_EPA', 'MAD/MAE'], ['MP_Ehull', 'MAD/MAE'], ['MP_Efermi', 'MAD/MAE'],
#         ['MP_Density', 'MAD/MAE'], ['MP_DensityAtomic', 'MAD/MAE'], ['MP_Volume', 'MAD/MAE'],
#         ['JARVISDFT_FEPA', 'MAD/MAE'], ['JARVISDFT_Bandgap_OPT', 'MAD/MAE'], ['JARVISDFT_TotEn', 'MAD/MAE'],
#         ['JARVISDFT_Ehull', 'MAD/MAE'], ['JARVISDFT_Bandgap_MBJ', 'MAD/MAE'], ['JARVISDFT_Kv', 'MAD/MAE'],
#         ['JARVISDFT_Gv', 'MAD/MAE'], ['JARVISDFT_SLME', 'MAD/MAE'], ['JARVISDFT_Spillage', 'MAD/MAE'],
#         ['JARVISDFT_Epsx_OPT', 'MAD/MAE'], ['JARVISDFT_Dielectric_DFPT', 'MAD/MAE'],
#         ['JARVISDFT_Max_Piezo_dij', 'MAD/MAE'], ['JARVISDFT_Max_Piezo_eij', 'MAD/MAE'],
#         ['JARVISDFT_MaxEFG', 'MAD/MAE'], ['JARVISDFT_ExfEn', 'MAD/MAE'], ['JARVISDFT_AvgMe', 'MAD/MAE'],
#         ['JARVISDFT_nSeebeck', 'MAD/MAE'], ['JARVISDFT_nPF', 'MAD/MAE'], ['JARVISDFT_pSeebeck', 'MAD/MAE'],
#         ['JARVISDFT_pPF', 'MAD/MAE'], ['SNUMAT_Bandgap_GGA', 'MAD/MAE'], ['SNUMAT_Bandgap_HSE', 'MAD/MAE'],
#         ['SNUMAT_Bandgap_GGA_Optical', 'MAD/MAE'], ['SNUMAT_Bandgap_HSE_Optical', 'MAD/MAE'],
#         ['GNoME_FEPA', 'MAD/MAE'], ['GNoME_DEPA', 'MAD/MAE'], ['GNoME_Bandgap', 'MAD/MAE'],
#         ['GNoME_TotEn', 'MAD/MAE'], ['GNoME_Volume', 'MAD/MAE'], ['GNoME_Density', 'MAD/MAE'],
#         ['hMOF_MaxCO2', 'MAD/MAE'], ['hMOF_MinCO2', 'MAD/MAE'], ['hMOF_LCD', 'MAD/MAE'],
#         ['hMOF_PLD', 'MAD/MAE'], ['hMOF_VoidFraction', 'MAD/MAE'], ['hMOF_SA_m2g', 'MAD/MAE'],
#         ['hMOF_SA_m2cm3', 'MAD/MAE'], ['Cantor_HEA_FEPA', 'MAD/MAE'], ['Cantor_HEA_EPA', 'MAD/MAE'],
#         ['Cantor_HEA_Ehull', 'MAD/MAE'], ['Cantor_HEA_VPA', 'MAD/MAE'], ['QMOF_TotEn', 'MAD/MAE'],
#         ['QMOF_Bandgap', 'MAD/MAE'], ['QMOF_LCD', 'MAD/MAE'], ['QMOF_PLD', 'MAD/MAE'],
#         ['JARVISQETB_EPA', 'MAD/MAE'], ['JARVISQETB_IndirBandgap', 'MAD/MAE'],
#         ['JARVISQETB_FEPA', 'MAD/MAE'], ['JARVISQETB_TotEn', 'MAD/MAE'], ['OQMD_Bandgap', 'MAD/MAE'],
#         ['OQMD_FEPA', 'MAD/MAE'], ['OMDB_Bandgap', 'MAD/MAE'],
#         ['composition_to_material_generation', 'smact_validity_ratio_in_all_%'],
#         ['bulk_modulus_to_material_generation', 'smact_validity_ratio_in_all_%'],
#         ['mol_instruction_chemical_disease_interaction_extraction', 'f1'],
#         ['mol_instruction_chemical_entity_recognition', 'f1'],
#         ['mol_instruction_chemical_protein_interaction_extraction', 'f1'],
#         ['mol_instruction_multi_choice_question', 'accuracy'],
#         ['mol_instruction_open_question', 'bert_score'],
#         ['mol_instruction_true_or_false_question', 'accuracy'],
#         ['mol_instruction_property_prediction_str', 'mae'],
#         ['mol_instruction_description_guided_molecule_design', 'exact_match_score'],
#         ['mol_instruction_forward_reaction_prediction', 'exact_match_score'],
#         ['mol_instruction_retrosynthesis', 'exact_match_score'],
#         ['mol_instruction_reagent_prediction', 'exact_match_score'],
#         ['mol_instruction_molecular_description_generation', 'rougeL'],
#         ['mol_instruction_catalytic_activity', 'rougeL'], ['mol_instruction_domain_motif', 'rougeL'],
#         ['mol_instruction_general_function', 'rougeL'], ['mol_instruction_protein_function', 'rougeL'],
#         ['mol_instruction_protein_design', 'Max SW score'], ['EC_number_CLEAN_EC_number_new', 'Accuracy'],
#         ['EC_number_CLEAN_EC_number_price', 'Accuracy'], ['Fold_type_fold_type', 'Accuracy'],
#         ['Function_CASPSimilarSeq_function', 'ROUGE-L'], ['Function_IDFilterSeq_function', 'ROUGE-L'],
#         ['Function_UniProtSeq_function', 'ROUGE-L'], ['gName2Cancer_gene_name_to_cancer', 'F1 Score'],
#         ['GO_CASPSimilarSeq_go', 'F1 Score'], ['GO_IDFilterSeq_go', 'F1 Score'], ['GO_UniProtSeq_go', 'F1 Score'],
#         ['gSymbol2Cancer_gene_symbol_to_cancer', 'F1 Score'], ['gSymbol2Tissue_gene_symbol_to_tissue', 'F1 Score'],
#         ['Keywords_CASPSimilarSeq_keywords', 'F1 Score'], ['Keywords_IDFilterSeq_keywords', 'F1 Score'],
#         ['Keywords_UniProtSeq_keywords', 'F1 Score'], ['Subcellular_localization_subcell_loc', 'Accuracy'],
#         ['solubility', 'accuracy'], ['stability', 'accuracy'], ['human_ppi', 'accuracy'], ['yeast_ppi', 'accuracy'],
#         ['unconditional_material_generation', 'smact_validity_ratio_in_all'],
#         ['unconditional_RNA_generation', 'average_mfe'], ['unconditional_protein_generation', 'valid_rate'],
#         ['unconditional_molecule_generation', 'validity']
#     ]
# )

# mini set summarizer
summarizer = dict(
    dataset_abbrs=[
        ['antibody_antigen-mini', 'MCC'], ['rna_protein_interaction-mini', 'MCC'], ['emp-mini', 'MCC'],
        ['enhancer_activity-mini', 'PCC'], ['tf_m-mini', 'MCC'], ['Isoform-mini', 'R2'], ['Modification-mini', 'AUC'],
        ['MeanRibosomeLoading-mini', 'R2'], ['ProgrammableRNASwitches-mini', 'R2'], ['CRISPROnTarget-mini', 'spearman'],
        ['promoter_enhancer_interaction-mini', 'MCC'], ['sirnaEfficiency-mini', 'mixed_score'], ['cpd-mini', 'MCC'],
        ['pd-mini', 'MCC'], ['tf_h-mini', 'MCC'],
        ['cpd-prom_core_all-mini', 'matthews_correlation_all'],
        ['cpd-prom_core_notata-mini', 'matthews_correlation_all'],
        ['cpd-prom_core_tata-mini', 'matthews_correlation_all'], ['pd-prom_300_all-mini', 'matthews_correlation_all'],
        ['pd-prom_300_notata-mini', 'matthews_correlation_all'], ['pd-prom_300_tata-mini', 'matthews_correlation_all'],
        ['tf-h-0-mini', 'matthews_correlation_all'], ['tf-h-1-mini', 'matthews_correlation_all'],
        ['tf-h-2-mini', 'matthews_correlation_all'], ['tf-h-3-mini', 'matthews_correlation_all'],
        ['tf-h-4-mini', 'matthews_correlation_all'], ['smol_forward_synthesis-mini', 'top1_exact_match'],
        ['smol_retrosynthesis-mini', 'top1_exact_match'], ['smol_molecule_captioning-mini', 'meteor_score'],
        ['smol_molecule_generation-mini', 'top1_exact_match'], ['smol_name_conversion-i2f-mini', 'top1_ele_match'],
        ['smol_name_conversion-i2s-mini', 'top1_exact_match'], ['smol_name_conversion-s2f-mini', 'top1_ele_match'],
        ['smol_name_conversion-s2i-mini', 'top1_split_match'], ['smol_property_prediction-esol-mini', 'RMSE'],
        ['smol_property_prediction-lipo-mini', 'RMSE'], ['smol_property_prediction-bbbp-mini', 'accuracy'],
        ['smol_property_prediction-clintox-mini', 'accuracy'], ['smol_property_prediction-hiv-mini', 'accuracy'],
        ['smol_property_prediction-sider-mini', 'accuracy'], ['retrosynthesis_USPTO_50K-mini', 'Top-1 Accuracy'],
        ['MP_IsStable-mini', 'AUC'], ['MP_IsGapDirect-mini', 'AUC'], ['SNUMAT_IsDirect-mini', 'AUC'],
        ['SNUMAT_IsDirect_HSE-mini', 'AUC'], ['SNUMAT_SOC-mini', 'AUC'], ['MP_FEPA-mini', 'MAD/MAE'],
        ['MP_Bandgap-mini', 'MAD/MAE'], ['MP_EPA-mini', 'MAD/MAE'], ['MP_Ehull-mini', 'MAD/MAE'], ['MP_Efermi-mini', 'MAD/MAE'],
        ['MP_Density-mini', 'MAD/MAE'], ['MP_DensityAtomic-mini', 'MAD/MAE'], ['MP_Volume-mini', 'MAD/MAE'],
        ['JARVISDFT_FEPA-mini', 'MAD/MAE'], ['JARVISDFT_Bandgap_OPT-mini', 'MAD/MAE'], ['JARVISDFT_TotEn-mini', 'MAD/MAE'],
        ['JARVISDFT_Ehull-mini', 'MAD/MAE'], ['JARVISDFT_Bandgap_MBJ-mini', 'MAD/MAE'], ['JARVISDFT_Kv-mini', 'MAD/MAE'],
        ['JARVISDFT_Gv-mini', 'MAD/MAE'], ['JARVISDFT_SLME-mini', 'MAD/MAE'], ['JARVISDFT_Spillage-mini', 'MAD/MAE'],
        ['JARVISDFT_Epsx_OPT-mini', 'MAD/MAE'], ['JARVISDFT_Dielectric_DFPT-mini', 'MAD/MAE'],
        ['JARVISDFT_Max_Piezo_dij-mini', 'MAD/MAE'], ['JARVISDFT_Max_Piezo_eij-mini', 'MAD/MAE'],
        ['JARVISDFT_MaxEFG-mini', 'MAD/MAE'], ['JARVISDFT_ExfEn-mini', 'MAD/MAE'], ['JARVISDFT_AvgMe-mini', 'MAD/MAE'],
        ['JARVISDFT_nSeebeck-mini', 'MAD/MAE'], ['JARVISDFT_nPF-mini', 'MAD/MAE'], ['JARVISDFT_pSeebeck-mini', 'MAD/MAE'],
        ['JARVISDFT_pPF-mini', 'MAD/MAE'], ['SNUMAT_Bandgap_GGA-mini', 'MAD/MAE'], ['SNUMAT_Bandgap_HSE-mini', 'MAD/MAE'],
        ['SNUMAT_Bandgap_GGA_Optical-mini', 'MAD/MAE'], ['SNUMAT_Bandgap_HSE_Optical-mini', 'MAD/MAE'],
        ['GNoME_FEPA-mini', 'MAD/MAE'], ['GNoME_DEPA-mini', 'MAD/MAE'], ['GNoME_Bandgap-mini', 'MAD/MAE'],
        ['GNoME_TotEn-mini', 'MAD/MAE'], ['GNoME_Volume-mini', 'MAD/MAE'], ['GNoME_Density-mini', 'MAD/MAE'],
        ['hMOF_MaxCO2-mini', 'MAD/MAE'], ['hMOF_MinCO2-mini', 'MAD/MAE'], ['hMOF_LCD-mini', 'MAD/MAE'],
        ['hMOF_PLD-mini', 'MAD/MAE'], ['hMOF_VoidFraction-mini', 'MAD/MAE'], ['hMOF_SA_m2g-mini', 'MAD/MAE'],
        ['hMOF_SA_m2cm3-mini', 'MAD/MAE'], ['Cantor_HEA_FEPA-mini', 'MAD/MAE'], ['Cantor_HEA_EPA-mini', 'MAD/MAE'],
        ['Cantor_HEA_Ehull-mini', 'MAD/MAE'], ['Cantor_HEA_VPA-mini', 'MAD/MAE'], ['QMOF_TotEn-mini', 'MAD/MAE'],
        ['QMOF_Bandgap-mini', 'MAD/MAE'], ['QMOF_LCD-mini', 'MAD/MAE'], ['QMOF_PLD-mini', 'MAD/MAE'],
        ['JARVISQETB_EPA-mini', 'MAD/MAE'], ['JARVISQETB_IndirBandgap-mini', 'MAD/MAE'],
        ['JARVISQETB_FEPA-mini', 'MAD/MAE'], ['JARVISQETB_TotEn-mini', 'MAD/MAE'], ['OQMD_Bandgap-mini', 'MAD/MAE'],
        ['OQMD_FEPA-mini', 'MAD/MAE'], ['OMDB_Bandgap-mini', 'MAD/MAE'],
        ['composition_to_material_generation-mini', 'smact_validity_ratio_in_all_%'],
        ['bulk_modulus_to_material_generation-mini', 'smact_validity_ratio_in_all_%'],
        ['mol_instruction_chemical_disease_interaction_extraction-mini', 'f1'],
        ['mol_instruction_chemical_entity_recognition-mini', 'f1'],
        ['mol_instruction_chemical_protein_interaction_extraction-mini', 'f1'],
        ['mol_instruction_multi_choice_question-mini', 'accuracy'],
        ['mol_instruction_open_question-mini', 'bert_score'],
        ['mol_instruction_true_or_false_question-mini', 'accuracy'],
        ['mol_instruction_property_prediction_str-mini', 'mae'],
        ['mol_instruction_description_guided_molecule_design-mini', 'exact_match_score'],
        ['mol_instruction_forward_reaction_prediction-mini', 'exact_match_score'],
        ['mol_instruction_retrosynthesis-mini', 'exact_match_score'],
        ['mol_instruction_reagent_prediction-mini', 'exact_match_score'],
        ['mol_instruction_molecular_description_generation-mini', 'rougeL'],
        ['mol_instruction_catalytic_activity-mini', 'rougeL'], ['mol_instruction_domain_motif-mini', 'rougeL'],
        ['mol_instruction_general_function-mini', 'rougeL'], ['mol_instruction_protein_function-mini', 'rougeL'],
        ['mol_instruction_protein_design-mini', 'Max SW score'], ['EC_number_CLEAN_EC_number_new-mini', 'Accuracy'],
        ['EC_number_CLEAN_EC_number_price-mini', 'Accuracy'], ['Fold_type_fold_type-mini', 'Accuracy'],
        ['Function_CASPSimilarSeq_function-mini', 'ROUGE-L'], ['Function_IDFilterSeq_function-mini', 'ROUGE-L'],
        ['Function_UniProtSeq_function-mini', 'ROUGE-L'], ['gName2Cancer_gene_name_to_cancer-mini', 'F1 Score'],
        ['GO_CASPSimilarSeq_go-mini', 'F1 Score'], ['GO_IDFilterSeq_go-mini', 'F1 Score'], ['GO_UniProtSeq_go-mini', 'F1 Score'],
        ['gSymbol2Cancer_gene_symbol_to_cancer-mini', 'F1 Score'], ['gSymbol2Tissue_gene_symbol_to_tissue-mini', 'F1 Score'],
        ['Keywords_CASPSimilarSeq_keywords-mini', 'F1 Score'], ['Keywords_IDFilterSeq_keywords-mini', 'F1 Score'],
        ['Keywords_UniProtSeq_keywords-mini', 'F1 Score'], ['Subcellular_localization_subcell_loc-mini', 'Accuracy'],
        ['solubility-mini', 'accuracy'], ['stability-mini', 'accuracy'], ['human_ppi-mini', 'accuracy'], ['yeast_ppi-mini', 'accuracy'],
        ['unconditional_material_generation-mini', 'smact_validity_ratio_in_all'],
        ['unconditional_RNA_generation-mini', 'average_mfe'], ['unconditional_protein_generation-mini', 'valid_rate'],
        ['unconditional_molecule_generation-mini', 'validity']
    ]
)