from mmengine.config import read_base

from opencompass.models import (HuggingFacewithChatTemplate,
                                TurboMindModelwithChatTemplate)
from opencompass.utils.text_postprocessors import extract_non_reasoning_content

with read_base():
    # Datasets
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

    from ...rjob import eval, infer  # noqa: F401, E501

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

hf_model = dict(type=HuggingFacewithChatTemplate,
                abbr='qwen-3-8b-hf-fullbench',
                path='Qwen/Qwen3-8B',
                max_out_len=8192,
                batch_size=8,
                run_cfg=dict(num_gpus=1),
                pred_postprocessor=dict(type=extract_non_reasoning_content))

tm_model = dict(type=TurboMindModelwithChatTemplate,
                abbr='qwen-3-8b-fullbench',
                path='Qwen/Qwen3-8B',
                engine_config=dict(session_len=32768, max_batch_size=1, tp=1),
                gen_config=dict(do_sample=False, enable_thinking=True),
                max_seq_len=32768,
                max_out_len=32768,
                batch_size=1,
                run_cfg=dict(num_gpus=1),
                pred_postprocessor=dict(type=extract_non_reasoning_content))

models = [hf_model, tm_model]

models = sorted(models, key=lambda x: x['run_cfg']['num_gpus'])

obj_judge_model = dict(
    type=TurboMindModelwithChatTemplate,
    abbr='qwen-3-8b-fullbench',
    path='Qwen/Qwen3-8B',
    engine_config=dict(session_len=46000, max_batch_size=1, tp=1),
    gen_config=dict(do_sample=False, enable_thinking=True),
    max_seq_len=46000,
    max_out_len=46000,
    batch_size=1,
    run_cfg=dict(num_gpus=1),
    pred_postprocessor=dict(type=extract_non_reasoning_content))

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
