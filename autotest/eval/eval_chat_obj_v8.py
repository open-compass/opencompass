from mmengine.config import read_base

with read_base():
    # Datasets
    from autotest.eval.models import interns1_models, judge_models
    from opencompass.configs.datasets.biodata.biodata_task_gen import \
        biodata_task_datasets  # noqa: F401, E501
    from opencompass.configs.datasets.MolInstructions_chem.mol_instructions_chem_gen import \
        mol_gen_selfies_datasets  # noqa: F401, E501
    from opencompass.configs.datasets.SciReasoner.scireasoner_gen import (  # noqa: F401, E501
        mini_bio_instruction_datasets, mini_composition_material_datasets,
        mini_GUE_datasets, mini_LLM4Mat_datasets,
        mini_modulus_material_datasets, mini_mol_biotext_datasets,
        mini_mol_mol_datasets, mini_mol_protein_datasets, mini_opi_datasets,
        mini_PEER_datasets, mini_Retrosynthesis_uspto50k_datasets,
        mini_smol_datasets, mini_UMG_Datasets, mini_uncond_material_datasets,
        mini_uncond_protein_datasets, mini_uncond_RNA_datasets)

datasets = [
    v[0] for k, v in locals().items()
    if k.endswith('_datasets') and isinstance(v, list) and len(v) > 0
    and 'mol' not in k and 'smol' not in k
]

datasets += [
    x for x in mini_mol_mol_datasets if 'property_prediction_str' in x['abbr']
    or 'description_guided_molecule_design' in x['abbr']
    or 'molecular_description_generation' in x['abbr']
]
datasets += [x for x in mini_mol_protein_datasets if 'protein' in x['abbr']]
datasets += [
    x for x in mini_opi_datasets
    if 'EC_number_CLEAN_EC_number_new' in x['abbr']
    or 'Subcellular_localization_subcell_loc' in x['abbr']
    or 'Fold_type_fold_type' in x['abbr']
    or 'Function_CASPSimilarSeq_function' in x['abbr']
]
datasets += [
    x for x in mini_smol_datasets
    if 'forward_synthesis' in x['abbr'] or 'retrosynthesis' in x['abbr']
    or 'molecule_captioning' in x['abbr'] or 'name_conversion-i2f' in x['abbr']
    or 'name_conversion-s2i' in x['abbr'] or 'property_prediction-esol' in
    x['abbr'] or 'property_prediction-bbbp' in x['abbr']
]

models = interns1_models

datasets = sum((v for k, v in locals().items() if k.endswith('_datasets')), [])

obj_judge_model = judge_models[0]

for d in datasets:
    if 'eval_cfg' in d and 'evaluator' in d['eval_cfg']:
        if 'atlas' in d['abbr'] and 'judge_cfg' in d['eval_cfg']['evaluator']:
            d['eval_cfg']['evaluator']['judge_cfg'] = dict(
                judgers=[obj_judge_model])
        elif 'judge_cfg' in d['eval_cfg']['evaluator']:
            d['eval_cfg']['evaluator']['judge_cfg'] = obj_judge_model
        elif 'llm_evaluator' in d['eval_cfg'][
                'evaluator'] and 'judge_cfg' in d[  # noqa
                    'eval_cfg']['evaluator']['llm_evaluator']:  # noqa
            d['eval_cfg']['evaluator']['llm_evaluator'][
                'judge_cfg'] = obj_judge_model
