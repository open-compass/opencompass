from mmengine.config import read_base

with read_base():
    # Datasets
    from autotest.eval.models import judge_models, models
    from opencompass.configs.datasets.atlas.atlas_val_gen_b2d1b6 import \
        atlas_datasets  # noqa: F401, E501
    from opencompass.configs.datasets.biodata.biodata_task_gen import \
        biodata_task_datasets  # noqa: F401, E501
    from opencompass.configs.datasets.CMPhysBench.cmphysbench_gen import \
        cmphysbench_datasets  # noqa: F401, E501
    from opencompass.configs.datasets.IFBench.IFBench_gen import \
        ifbench_datasets  # noqa: F401, E501
    from opencompass.configs.datasets.livecodebench_pro.livecodebench_pro_gen import \
        lcb_pro_datasets  # noqa: F401, E501
    from opencompass.configs.datasets.MolInstructions_chem.mol_instructions_chem_gen import \
        mol_gen_selfies_datasets  # noqa: F401, E501
    from opencompass.configs.datasets.openswi.openswi_gen import \
        openswi_datasets  # noqa: F401, E501

models = models

datasets = sum((v for k, v in locals().items() if k.endswith('_datasets')), [])

for d in datasets:
    if 'n' in d:
        d['n'] = 1
    if 'reader_cfg' in d:
        d['reader_cfg']['test_range'] = '[0:4]'
    else:
        d['test_range'] = '[0:4]'
    if 'eval_cfg' in d and 'dataset_cfg' in d['eval_cfg'][
            'evaluator'] and 'reader_cfg' in d['eval_cfg']['evaluator'][
                'dataset_cfg']:
        d['eval_cfg']['evaluator']['dataset_cfg']['reader_cfg'][
            'test_range'] = '[0:4]'
    if 'eval_cfg' in d and 'llm_evaluator' in d['eval_cfg'][
            'evaluator'] and 'dataset_cfg' in d['eval_cfg']['evaluator'][
                'llm_evaluator']:
        d['eval_cfg']['evaluator']['llm_evaluator']['dataset_cfg'][
            'reader_cfg']['test_range'] = '[0:4]'

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
