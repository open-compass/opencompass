from mmengine.config import read_base

from opencompass.models import OpenAISDKRollout
from opencompass.utils.text_postprocessors import extract_non_reasoning_content

with read_base():
    from autotest.infer.chat_datasets import datasets

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

BASE_ROLLOUT = dict(
    type=OpenAISDKRollout,
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

API_ROLLOUT_BASIC = dict(
    **BASE_ROLLOUT,
    abbr='lmdeploy-api-test-rollout',
    max_out_len=1024,
    max_seq_len=4096,
    temperature=0.01,
    logprobs=True,
    top_logprobs=5,
    extra_body=dict(top_k=20),
    openai_extra_kwargs=dict(top_p=0.95),
)

API_ROLLOUT_STOP = dict(
    **BASE_ROLLOUT,
    abbr='lmdeploy-api-test-rollout-stop',
    max_out_len=512,
    max_seq_len=4096,
    temperature=0.2,
    logprobs=True,
    top_logprobs=5,
    openai_extra_kwargs=dict(
        stop=[' and', '</think>', ' to', '\n\n', 'Question:', 'Answer:'],
        top_p=0.9,
    ),
)

API_ROLLOUT_COMBINE = dict(
    **BASE_ROLLOUT,
    abbr='lmdeploy-api-test-rollout-combine',
    max_out_len=512,
    max_seq_len=4096,
    temperature=0.2,
    logprobs=True,
    top_logprobs=5,
    openai_extra_kwargs=dict(
        presence_penalty=0.3,
        frequency_penalty=0.2,
        top_p=0.85,
        seed=42,
        user='opencompass-regression',
    ),
)

API_ROLLOUT_IGNORE_EOS = dict(
    **BASE_ROLLOUT,
    abbr='lmdeploy-api-test-rollout-ignore-eos',
    max_out_len=128,
    max_seq_len=4096,
    temperature=0.2,
    logprobs=True,
    top_logprobs=5,
    extra_body={
        'ignore_eos': True,
    },
)

API_ROLLOUT_NO_THINK = dict(
    **BASE_ROLLOUT,
    abbr='lmdeploy-api-test-rollout-no-think',
    max_out_len=128,
    max_seq_len=4096,
    temperature=0.2,
    logprobs=True,
    top_logprobs=5,
    extra_body={
        'enable_thinking': False,
    },
)

API_ROLLOUT_LONG_OUTPUT_128K = dict(
    **BASE_ROLLOUT,
    abbr='lmdeploy-api-test-rollout-long-output-128k',
    max_out_len=1024,
    max_seq_len=131072,
    temperature=0.01,
    logprobs=True,
    top_logprobs=5,
)

models = [
    API_ROLLOUT_BASIC,
    API_ROLLOUT_STOP,
    API_ROLLOUT_COMBINE,
    API_ROLLOUT_IGNORE_EOS,
    API_ROLLOUT_NO_THINK,
    API_ROLLOUT_LONG_OUTPUT_128K,
]

for m in models:
    if 'openai_extra_kwargs' not in m:
        m['openai_extra_kwargs'] = dict(do_sample=False)
    else:
        m['openai_extra_kwargs']['do_sample'] = False