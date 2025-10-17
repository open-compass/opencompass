from mmengine.config import read_base

with read_base():
    # Datasets
    from opencompass.configs.datasets.eese.eese_llm_judge_gen import \
        eese_datasets  # noqa: F401, E501
    from opencompass.configs.models.hf_internlm.hf_internlm3_8b_instruct import \
        models as hf_internlm3_8b_instruct_model  # noqa: F401, E501
    from opencompass.configs.models.hf_internlm.lmdeploy_internlm3_8b_instruct import \
        models as lmdeploy_internlm3_8b_instruct_model  # noqa: F401, E501

    from ...volc import infer  # noqa: F401, E501

datasets = [
    v[0] for k, v in locals().items()
    if k.endswith('_datasets') and isinstance(v, list) and len(v) > 0
]

for d in datasets:
    d['reader_cfg']['test_range'] = '[0:16]'
    if 'dataset_cfg' in d['eval_cfg']['evaluator'] and 'reader_cfg' in d[
            'eval_cfg']['evaluator']['dataset_cfg']:
        d['eval_cfg']['evaluator']['dataset_cfg']['reader_cfg'][
            'test_range'] = '[0:16]'
    if 'llm_evaluator' in d['eval_cfg']['evaluator'] and 'dataset_cfg' in d[
            'eval_cfg']['evaluator']['llm_evaluator']:
        d['eval_cfg']['evaluator']['llm_evaluator']['dataset_cfg'][
            'reader_cfg']['test_range'] = '[0:16]'

models = sum([v for k, v in locals().items() if k.endswith('_model')], [])
for m in models:
    m['abbr'] = m['abbr'] + '_fullbench'
    if 'turbomind' in m['abbr'] or 'lmdeploy' in m['abbr']:
        m['engine_config']['max_batch_size'] = 1
        m['batch_size'] = 1

models = sorted(models, key=lambda x: x['run_cfg']['num_gpus'])

obj_judge_model = lmdeploy_internlm3_8b_instruct_model[0]
obj_judge_model['engine_config']['max_batch_size'] = 1
obj_judge_model['engine_config']['cache_max_entry_count'] = 0.6
obj_judge_model['batch_size'] = 1

for d in datasets:
    if 'judge_cfg' in d['eval_cfg']['evaluator']:
        d['eval_cfg']['evaluator']['judge_cfg'] = obj_judge_model
    if 'llm_evaluator' in d['eval_cfg']['evaluator'] and 'judge_cfg' in d[
            'eval_cfg']['evaluator']['llm_evaluator']:
        d['eval_cfg']['evaluator']['llm_evaluator'][
            'judge_cfg'] = obj_judge_model
