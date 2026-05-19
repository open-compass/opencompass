import os

from opencompass.models import OpenAISDK
from opencompass.partitioners import NumWorkerPartitioner
from opencompass.runners import LocalRunner
from opencompass.tasks import OpenICLInferConcurrentTask, OpenICLInferTask
from opencompass.utils.text_postprocessors import extract_non_reasoning_content

API_BASE = 'http://localhost:26333/v1'
MODEL_PATH = 'mock_test'
TOKENIZER_PATH = 'Qwen/Qwen3-8B'

models = [
    dict(
        abbr='mock_test',
        type=OpenAISDK,
        key='EMPTY',
        openai_api_base=API_BASE,
        path=MODEL_PATH,
        # tokenizer_path=TOKENIZER_PATH,
        rpm_verbose=True,
        meta_template=dict(round=[
            dict(role='SYSTEM', api_role='SYSTEM'),
            dict(role='HUMAN', api_role='HUMAN'),
            dict(role='BOT', api_role='BOT', generate=True),
        ]),
        temperature=0,
        max_workers=1024,
        max_seq_len=524288,
        mode='none',
        retry=3,
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
        # tokenizer_path=TOKENIZER_PATH,
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
        max_seq_len=524288,
        max_workers=1024,
        query_per_second=128,
        mode='none',
        retry=3,
        pred_postprocessor=dict(type=extract_non_reasoning_content),
    )
]

# NumWorkerPartitioner cache; {config.fullbench_version} is filled at runtime.
_dataset_size_root = os.environ.get('REPORT_DIR', '.').rstrip('/')
_dataset_type = os.environ.get('CHAT_TYPE', 'default').rstrip('/')
dataset_size_path = (f'{_dataset_size_root}/dataset_size_{_dataset_type}.json')

concurrent_infer = dict(
    partitioner=dict(
        type=NumWorkerPartitioner,
        num_worker=1,
        dataset_size_path=dataset_size_path,
    ),
    runner=dict(
        type=LocalRunner,
        max_num_workers=64,
        retry=0,
        task=dict(type=OpenICLInferConcurrentTask),
    ),
)

common_infer = dict(
    partitioner=dict(
        type=NumWorkerPartitioner,
        num_worker=1,
        dataset_size_path=dataset_size_path,
    ),
    runner=dict(type=LocalRunner,
                task=dict(type=OpenICLInferTask),
                max_num_workers=128),
)
