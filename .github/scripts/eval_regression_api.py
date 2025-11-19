from mmengine.config import read_base

from opencompass.models import TurboMindModelwithChatTemplate
from opencompass.models.openai_api import OpenAISDK
from opencompass.models.openai_streaming import OpenAISDKStreaming
from opencompass.utils.text_postprocessors import extract_non_reasoning_content

with read_base():
    # choose a list of datasets
    from opencompass.configs.datasets.gsm8k.gsm8k_gen import \
        gsm8k_datasets  # noqa: F401, E501
    from opencompass.configs.datasets.HLE.hle_llmverify_academic import \
        hle_datasets  # noqa: F401, E501
    from opencompass.configs.datasets.IFEval.IFEval_gen_353ae7 import \
        ifeval_datasets  # noqa: F401, E501
    from opencompass.configs.datasets.mmlu_pro.mmlu_pro_0shot_cot_gen_08c1de import \
        mmlu_pro_datasets  # noqa: F401, E501
    from opencompass.configs.datasets.race.race_gen import \
        race_datasets  # noqa: F401, E501

mmlu_pro_datasets = [
    x for x in mmlu_pro_datasets if 'math' in x['abbr'] or 'other' in x['abbr']
]

datasets = sum((v for k, v in locals().items() if k.endswith('_datasets')), [])

api_meta_template = dict(
    round=[
        dict(role='HUMAN', api_role='HUMAN'),
        dict(role='BOT', api_role='BOT', generate=True),
    ],
    reserved_roles=[dict(role='SYSTEM', api_role='SYSTEM')],
)

models = [
    dict(abbr='lmdeploy-api-test',
         type=OpenAISDK,
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
         pred_postprocessor=dict(type=extract_non_reasoning_content)),
    dict(abbr='lmdeploy-api-streaming-test',
         type=OpenAISDKStreaming,
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
         stream=True,
         retry=20,
         pred_postprocessor=dict(type=extract_non_reasoning_content)),
    dict(
        abbr='lmdeploy-api-streaming-test-chunk',
        type=OpenAISDKStreaming,
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
        stream=True,
        retry=20,
        pred_postprocessor=dict(type=extract_non_reasoning_content),
        stream_chunk_size=10,
        verbose=True,
    ),
    dict(abbr='lmdeploy-api-test-maxlen',
         type=OpenAISDK,
         key='EMPTY',
         openai_api_base='http://localhost:23333/v1',
         path='Qwen/Qwen3-8B',
         tokenizer_path='Qwen/Qwen3-8B',
         rpm_verbose=True,
         meta_template=api_meta_template,
         query_per_second=128,
         max_out_len=8092,
         max_seq_len=8092,
         temperature=0.01,
         batch_size=128,
         retry=20,
         pred_postprocessor=dict(type=extract_non_reasoning_content)),
    dict(abbr='lmdeploy-api-test-maxlen-mid',
         type=OpenAISDK,
         key='EMPTY',
         openai_api_base='http://localhost:23333/v1',
         path='Qwen/Qwen3-8B',
         tokenizer_path='Qwen/Qwen3-8B',
         rpm_verbose=True,
         meta_template=api_meta_template,
         query_per_second=128,
         max_out_len=8092,
         max_seq_len=8092,
         temperature=0.01,
         batch_size=128,
         retry=20,
         mode='mid',
         pred_postprocessor=dict(type=extract_non_reasoning_content)),
    dict(abbr='lmdeploy-api-test-nothink',
         type=OpenAISDK,
         key='EMPTY',
         openai_api_base='http://localhost:23333/v1',
         path='Qwen/Qwen3-8B',
         tokenizer_path='Qwen/Qwen3-8B',
         rpm_verbose=True,
         meta_template=api_meta_template,
         query_per_second=128,
         max_out_len=8092,
         max_seq_len=8092,
         temperature=0.01,
         batch_size=128,
         retry=20,
         openai_extra_kwargs={
             'top_p': 0.95,
         },
         extra_body={'chat_template_kwargs': {
             'enable_thinking': False
         }},
         pred_postprocessor=dict(type=extract_non_reasoning_content)),
    dict(
        abbr='lmdeploy-api-test-chat-template',
        type=OpenAISDK,
        key='EMPTY',
        openai_api_base='http://localhost:23333/v1',
        path='Qwen/Qwen3-8B',
        tokenizer_path='Qwen/Qwen3-8B',
        rpm_verbose=True,
        meta_template=dict(begin=dict(role='SYSTEM',
                                      api_role='SYSTEM',
                                      prompt='you are a helpful AI.'),
                           round=[
                               dict(role='HUMAN', api_role='HUMAN'),
                               dict(role='BOT', api_role='BOT', generate=True),
                           ]),
        query_per_second=128,
        max_out_len=1024,
        max_seq_len=1024,
        temperature=0.01,
        batch_size=128,
        retry=20,
        openai_extra_kwargs={
            'top_p': 0.95,
        },
        extra_body={'chat_template_kwargs': {
            'enable_thinking': False
        }},
        pred_postprocessor=dict(type=extract_non_reasoning_content),
    ),
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

core_summary_groups = [
    {
        'name':
        'core_average',
        'subsets': [
            ['IFEval', 'Prompt-level-strict-accuracy'],
            ['hle_llmjudge', 'accuracy'],
            ['mmlu_pro', 'naive_average'],
            'mmlu_pro_math',
            'mmlu_pro_other',
            'gsm8k',
            'race-middle',
            'race-high',
        ],
    },
]

summarizer = dict(
    dataset_abbrs=[
        ['core_average', 'naive_average'],
        ['IFEval', 'Prompt-level-strict-accuracy'],
        ['hle_llmjudge', 'accuracy'],
        ['mmlu_pro', 'naive_average'],
        'mmlu_pro_math',
        'mmlu_pro_other',
        'gsm8k',
        'race-middle',
        'race-high',
    ],
    summary_groups=sum(
        [v
         for k, v in locals().items() if k.endswith('_summary_groups')], []) +
    core_summary_groups,
)
