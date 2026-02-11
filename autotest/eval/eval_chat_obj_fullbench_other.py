from mmengine.config import read_base

with read_base():
    # Datasets
    from autotest.eval.models import judge_models, models
    from opencompass.configs.chatml_datasets.C_MHChem.C_MHChem_gen import \
        datasets as C_MHChem_chatml_datasets  # noqa: F401, E501
    from opencompass.configs.chatml_datasets.CPsyExam.CPsyExam_gen import \
        datasets as CPsyExam_chatml_datasets  # noqa: F401, E501
    from opencompass.configs.chatml_datasets.MaScQA.MaScQA_gen import \
        datasets as MaScQA_chatml_datasets  # noqa: F401, E501
    from opencompass.configs.chatml_datasets.UGPhysics.UGPhysics_gen import \
        datasets as UGPhysics_chatml_datasets  # noqa: F401, E501
    from opencompass.configs.datasets.eese.eese_llm_judge_gen import \
        eese_datasets  # noqa: F401, E501

models = models

chatml_datasets = [
    v[0] for k, v in locals().items()
    if k.endswith('_chatml_datasets') and isinstance(v, list) and len(v) > 0
]

datasets = [eese_datasets[0]]

for d in chatml_datasets:
    d['test_range'] = '[0:16]'

for d in datasets:
    if 'reader_cfg' in d:
        d['reader_cfg']['test_range'] = '[0:16]'
    else:
        d['test_range'] = '[0:16]'
    if 'eval_cfg' in d and 'dataset_cfg' in d['eval_cfg'][
            'evaluator'] and 'reader_cfg' in d['eval_cfg']['evaluator'][
                'dataset_cfg']:
        d['eval_cfg']['evaluator']['dataset_cfg']['reader_cfg'][
            'test_range'] = '[0:16]'
    if 'eval_cfg' in d and 'llm_evaluator' in d['eval_cfg'][
            'evaluator'] and 'dataset_cfg' in d['eval_cfg']['evaluator'][
                'llm_evaluator']:
        d['eval_cfg']['evaluator']['llm_evaluator']['dataset_cfg'][
            'reader_cfg']['test_range'] = '[0:16]'

obj_judge_model = judge_models[0]

for d in datasets:
    if 'eval_cfg' in d and 'evaluator' in d['eval_cfg']:
        if 'judge_cfg' in d['eval_cfg']['evaluator']:
            d['eval_cfg']['evaluator']['judge_cfg'] = obj_judge_model
        if 'llm_evaluator' in d['eval_cfg']['evaluator'] and 'judge_cfg' in d[
                'eval_cfg']['evaluator']['llm_evaluator']:
            d['eval_cfg']['evaluator']['llm_evaluator'][
                'judge_cfg'] = obj_judge_model

for d in chatml_datasets:
    if 'judge_cfg' in d['evaluator']:
        d['evaluator']['judge_cfg'] = obj_judge_model
    if 'llm_evaluator' in d['evaluator'] and 'judge_cfg' in d['evaluator'][
            'llm_evaluator']:
        d['evaluator']['llm_evaluator']['judge_cfg'] = obj_judge_model
