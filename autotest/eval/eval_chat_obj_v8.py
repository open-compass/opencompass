from mmengine.config import read_base

with read_base():
    # Datasets
    from autotest.eval.models import judge_models, test_models
    from opencompass.configs.chatml_datasets.AMO_Bench.AMO_Bench_gen import \
        datasets as AMO_Bench_chatml  # noqa: F401, E501
    from opencompass.configs.chatml_datasets.C_MHChem.C_MHChem_gen import \
        datasets as C_MHChem_chatml  # noqa: F401, E501
    from opencompass.configs.chatml_datasets.CPsyExam.CPsyExam_gen import \
        datasets as CPsyExam_chatml  # noqa: F401, E501
    from opencompass.configs.chatml_datasets.CS_Bench.CS_Bench_gen import \
        datasets as CS_Bench_chatml  # noqa: F401, E501
    from opencompass.configs.chatml_datasets.HMMT2025.HMMT2025_gen import \
        datasets as HMMT2025_chatml  # noqa: F401, E501
    from opencompass.configs.chatml_datasets.IMO_Bench_AnswerBench.IMO_Bench_AnswerBench_gen import \
        datasets as IMO_Bench_AnswerBench_chatml  # noqa: F401, E501
    from opencompass.configs.chatml_datasets.MaScQA.MaScQA_gen import \
        datasets as MaScQA_chatml  # noqa: F401, E501
    from opencompass.configs.chatml_datasets.UGD_hard.UGD_hard_gen import \
        datasets as UGD_hard_chatml  # noqa: F401, E501
    from opencompass.configs.chatml_datasets.UGPhysics.UGPhysics_gen import \
        datasets as UGPhysics_chatml  # noqa: F401, E501
    from opencompass.configs.datasets.aime2026.aime2026_cascade_eval_gen_6ff468 import \
        aime2026_datasets  # noqa: F401, E501
    from opencompass.configs.datasets.biodata.biodata_task_gen import \
        biodata_task_datasets  # noqa: F401, E501
    from opencompass.configs.datasets.eese.eese_llm_judge_gen import \
        eese_datasets  # noqa: F401, E501
    from opencompass.configs.datasets.hmmt2026.hmmt2026_cascade_eval_gen_6ff468 import \
        hmmt2026_datasets  # noqa: F401, E501
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

models = test_models

datasets = sum((v for k, v in locals().items() if k.endswith('_datasets')), [])

chatml_datasets = [
    v[0] for k, v in locals().items()
    if k.endswith('_chatml_datasets') and isinstance(v, list) and len(v) > 0
]

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

for dataset in chatml_datasets:
    if dataset['evaluator']['type'] == 'llm_evaluator':
        dataset['evaluator']['judge_cfg'] = obj_judge_model
    if dataset['evaluator']['type'] == 'cascade_evaluator':
        dataset['evaluator']['llm_evaluator']['judge_cfg'] = obj_judge_model
