from opencompass.models import OpenAISDK
from opencompass.utils.text_postprocessors import extract_non_reasoning_content

API_BASE = 'http://localhost:23333/v1'
MODEL_PATH = 'Qwen/Qwen3-8B'
TOKENIZER_PATH = 'Qwen/Qwen3-8B'

api_meta_template = dict(round=[
    dict(role='SYSTEM', api_role='SYSTEM'),
    dict(role='HUMAN', api_role='HUMAN'),
    dict(role='BOT', api_role='BOT', generate=True),
]),

models = [
    dict(
        type=OpenAISDK,
        key='EMPTY',
        openai_api_base=API_BASE,
        path=MODEL_PATH,
        tokenizer_path=TOKENIZER_PATH,
        rpm_verbose=True,
        meta_template=api_meta_template,
        max_seq_len=4096,
        query_per_second=128,
        temperature=0,
        max_worker=128,
        mode='mid',
        retry=20,
        pred_postprocessor=dict(type=extract_non_reasoning_content),
    )
]
