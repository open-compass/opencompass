from opencompass.models import OpenAISDK
from opencompass.partitioners import NumWorkerPartitioner
from opencompass.runners import LocalRunner
from opencompass.tasks import OpenICLInferConcurrentTask, OpenICLInferTask
from opencompass.utils.text_postprocessors import extract_non_reasoning_content

API_BASE = 'http://localhost:23333/v1'
MODEL_PATH = 'Qwen/Qwen3-8B'
TOKENIZER_PATH = 'Qwen/Qwen3-8B'

models = [
    dict(
        abbr='mock_test',
        type=OpenAISDK,
        key='EMPTY',
        openai_api_base=API_BASE,
        path=MODEL_PATH,
        tokenizer_path=TOKENIZER_PATH,
        rpm_verbose=True,
        meta_template=dict(round=[
            dict(role='SYSTEM', api_role='SYSTEM'),
            dict(role='HUMAN', api_role='HUMAN'),
            dict(role='BOT', api_role='BOT', generate=True),
        ]),
        temperature=0,
        max_workers=1024,
        mode='mid',
        retry=20,
        pred_postprocessor=dict(type=extract_non_reasoning_content),
    )
]

raw_template_models = [
    dict(
        abbr='raw_template_mock_test',
        type=OpenAISDK,
        key='EMPTY',
        openai_api_base=API_BASE,
        path=MODEL_PATH,
        tokenizer_path=TOKENIZER_PATH,
        rpm_verbose=True,
        meta_template=[{
            'content': 'Extra test system prompt.',
            'role': 'system'
        }, {
            'content': 'Extra test user prompt.',
            'role': 'user'
        }],
        temperature=0,
        batch_size=1024,
        max_workers=1024,
        mode='mid',
        retry=20,
        pred_postprocessor=dict(type=extract_non_reasoning_content),
    )
]

concurrent_infer = dict(
    partitioner=dict(type=NumWorkerPartitioner, num_worker=1),
    runner=dict(
        type=LocalRunner,
        max_num_workers=64,
        retry=0,
        task=dict(type=OpenICLInferConcurrentTask),
    ),
)

common_infer = dict(
    partitioner=dict(type=NumWorkerPartitioner, num_worker=1),
    runner=dict(type=LocalRunner,
                task=dict(type=OpenICLInferTask),
                max_num_workers=128),
)
