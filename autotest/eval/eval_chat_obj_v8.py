from mmengine.config import read_base

with read_base():
    # Datasets
    from autotest.eval.models import interns1_models, judge_models
    from opencompass.configs.datasets.biodata.biodata_task_gen import \
        biodata_task_datasets  # noqa: F401, E501
    from opencompass.configs.datasets.MolInstructions_chem.mol_instructions_chem_gen import \
        mol_gen_selfies_datasets  # noqa: F401, E501
    from opencompass.configs.datasets.SciReasoner.scireasoner_gen import \
        scireasoner_mini_datasets  # noqa: F401, E501

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
