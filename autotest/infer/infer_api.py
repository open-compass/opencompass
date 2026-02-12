from mmengine.config import read_base

from opencompass.models.openai_api import OpenAISDK
from opencompass.models.openai_streaming import OpenAISDKStreaming
from opencompass.utils.text_postprocessors import extract_non_reasoning_content

with read_base():
    from autotest.infer.base_datasets import datasets
    from autotest.infer.constant import meta_template as test_meta_template

datasets = datasets

api_meta_template = dict(
    round=[
        dict(role='HUMAN', api_role='HUMAN'),
        dict(role='BOT', api_role='BOT', generate=True),
    ],
    reserved_roles=[dict(role='SYSTEM', api_role='SYSTEM')],
)
API_BASE = 'http://localhost:23333/v1'
MODEL_PATH = 'Qwen/Qwen3-8B'
TOKENIZER_PATH = 'Qwen/Qwen3-8B'

BASE_API = dict(
    type=OpenAISDK,
    key='EMPTY',
    openai_api_base=API_BASE,
    path=MODEL_PATH,
    tokenizer_path=TOKENIZER_PATH,
    rpm_verbose=True,
    meta_template=api_meta_template,
    query_per_second=128,
    batch_size=128,
    retry=20,
    pred_postprocessor=dict(type=extract_non_reasoning_content),
)

BASE_STREAMING = dict(
    type=OpenAISDKStreaming,
    key='EMPTY',
    openai_api_base=API_BASE,
    path=MODEL_PATH,
    tokenizer_path=TOKENIZER_PATH,
    rpm_verbose=True,
    meta_template=api_meta_template,
    query_per_second=128,
    batch_size=128,
    stream=True,
    retry=20,
    pred_postprocessor=dict(type=extract_non_reasoning_content),
)

API_BASIC = dict(
    **BASE_API,
    abbr='lmdeploy-api-test',
    max_out_len=1024,
    max_seq_len=4096,
    temperature=0.01,
)

API_STREAMING = dict(
    **BASE_STREAMING,
    abbr='lmdeploy-api-streaming-test',
    max_out_len=1024,
    max_seq_len=4096,
    temperature=0.01,
)

API_STREAMING_CHUNK = dict(
    **BASE_STREAMING,
    abbr='lmdeploy-api-streaming-test-chunk',
    max_out_len=1024,
    max_seq_len=4096,
    temperature=0.01,
    stream_chunk_size=10,
    verbose=True,
)

API_MAXLEN = dict(
    **BASE_API,
    abbr='lmdeploy-api-test-maxlen',
    max_out_len=4096,
    max_seq_len=4096,
    temperature=0.01,
)

API_MAXLEN_MID = dict(
    **BASE_API,
    abbr='lmdeploy-api-test-maxlen-mid',
    max_out_len=4048,
    max_seq_len=4096,
    temperature=0.01,
    mode='mid',
)

API_NOTHINK = dict(
    **BASE_API,
    abbr='lmdeploy-api-test-nothink',
    max_out_len=4096,
    max_seq_len=4096,
    temperature=0.01,
    openai_extra_kwargs={
        'top_p': 0.95,
    },
    extra_body={'chat_template_kwargs': {
        'enable_thinking': False
    }},
)

API_IGNORE_EOS = dict(
    **BASE_API,
    abbr='lmdeploy-api-test-ignore-eos',
    max_out_len=128,
    max_seq_len=4096,
    temperature=0.2,
    extra_body={
        'ignore_eos': True,
    },
)

API_CHAT_TEMPLATE = dict(
    **BASE_API,
    abbr='lmdeploy-api-test-chat-template',
    max_out_len=1024,
    max_seq_len=1024,
    temperature=0.01,
    openai_extra_kwargs={
        'top_p': 0.95,
    },
    extra_body={'chat_template_kwargs': {
        'enable_thinking': False
    }},
)

API_CHAT_TEMPLATE['meta_template'] = test_meta_template

API_OPENAI_STOP = dict(
    **BASE_API,
    abbr='lmdeploy-api-test-openai-stop',
    max_out_len=512,
    max_seq_len=4096,
    temperature=0.2,
    openai_extra_kwargs={
        'stop': [' and', '</think>', ' to', '\n\n', 'Question:', 'Answer:'],
    },
)

API_OPENAI_LOGPROBS = dict(
    **BASE_API,
    abbr='lmdeploy-api-test-openai-logprobs',
    max_out_len=256,
    max_seq_len=4096,
    temperature=0.2,
    openai_extra_kwargs={
        'logprobs': True,
        'top_logprobs': 5,
    },
)

API_OPENAI_COMBINE = dict(
    **BASE_API,
    abbr='lmdeploy-api-test-openai-combine',
    max_out_len=512,
    max_seq_len=4096,
    temperature=0.2,
    openai_extra_kwargs={
        'presence_penalty': 0.3,
        'frequency_penalty': 0.2,
        'top_p': 0.85,
        'seed': 42,
        'user': 'opencompass-regression',
    },
)
API_LONG_OUTPUT_128K = dict(
    **BASE_API,
    abbr='lmdeploy-api-test-long-output-128k',
    max_out_len=4096,
    max_seq_len=131072,
    temperature=0.01,
)

models = [
    API_BASIC,
    API_STREAMING,
    API_STREAMING_CHUNK,
    API_MAXLEN,
    API_MAXLEN_MID,
    API_NOTHINK,
    API_IGNORE_EOS,
    API_CHAT_TEMPLATE,
    API_OPENAI_STOP,
    API_OPENAI_LOGPROBS,
    API_OPENAI_COMBINE,
    API_LONG_OUTPUT_128K,
]
