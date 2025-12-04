from mmengine.config import read_base

from opencompass.models import OpenAISDKRollout, TurboMindModelwithChatTemplate
from opencompass.utils.text_postprocessors import extract_non_reasoning_content

with read_base():
    choose a list of datasets
     from opencompass.configs.datasets.aime2025.aime2025_cascade_eval_gen_5e9f4f import \
         aime2025_datasets  # noqa: F401, E501

datasets = sum((v for k, v in locals().items() if k.endswith('_datasets')),
               [])

num_repeat = 2
for item in datasets:
    item['abbr'] += f'_rollout_rep{num_repeat}'
    item['n'] = num_repeat

api_meta_template = dict(
    round=[
        dict(role='HUMAN', api_role='HUMAN'),
        dict(role='BOT', api_role='BOT', generate=True),
    ],
    reserved_roles=[dict(role='SYSTEM', api_role='SYSTEM')],
)

models = [
    dict(abbr='lmdeploy-api-test-rollout',
         type=OpenAISDKRollout,
         key='EMPTY',
         openai_api_base='http://localhost:23333/v1',
         path='Qwen/Qwen3-8B',
         tokenizer_path='Qwen/Qwen3-8B',
         rpm_verbose=True,
         meta_template=api_meta_template,
         query_per_second=128,
         max_out_len=1024,
         max_seq_len=4096,
         temperature=0.01,
         batch_size=128,
         retry=20,
         extra_body=dict(
            top_k=20,
         ),
         openai_extra_kwargs=dict(
            top_p=0.95,
         ),
         pred_postprocessor=dict(type=extract_non_reasoning_content)),
]

obj_judge_model = dict(type=TurboMindModelwithChatTemplate,
                       abbr='qwen-3-8b-fullbench',
                       path='Qwen/Qwen3-8B',
                       engine_config=dict(session_len=46000,
                                          max_batch_size=1,
                                          tp=1),
                       gen_config=dict(do_sample=False, enable_thinking=False),
                       max_seq_len=46000,
                       max_out_len=46000,
                       batch_size=1,
                       run_cfg=dict(num_gpus=1))

for d in datasets:
    if 'judge_cfg' in d['eval_cfg']['evaluator']:
        d['eval_cfg']['evaluator']['judge_cfg'] = obj_judge_model
    if 'llm_evaluator' in d['eval_cfg']['evaluator'] and 'judge_cfg' in d[
            'eval_cfg']['evaluator']['llm_evaluator']:
        d['eval_cfg']['evaluator']['llm_evaluator'][
            'judge_cfg'] = obj_judge_model
