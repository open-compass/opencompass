from opencompass.models import OpenAISDKStreaming
from opencompass.utils.text_postprocessors import extract_non_reasoning_content

api_meta_template = dict(round=[
    dict(role='SYSTEM', api_role='system'),
    dict(role='HUMAN', api_role='user'),
    dict(role='BOT', api_role='assistant', generate=True),
])

models = [
    dict(
        type=OpenAISDKStreaming,
        abbr='intern-s1-pro-api',
        path='intern-s1-pro',
        key='INTERN_API_KEY',
        openai_api_base='https://chat.intern-ai.org.cn/api/v1',
        meta_template=api_meta_template,
        temperature=0.8,
        openai_extra_kwargs=dict(top_p=0.95),
        max_seq_len=262144,
        max_out_len=131072,
        query_per_second=1,
        retry=10,
        batch_size=8,
        max_workers=8,
        pred_postprocessor=dict(type=extract_non_reasoning_content),
    )
]
