from opencompass.models import OpenAISDK
from opencompass.utils.text_postprocessors import extract_non_reasoning_content

api_meta_template = dict(
    round=[
        dict(role='HUMAN', api_role='HUMAN'),
        dict(role='BOT', api_role='BOT', generate=True),
    ],
),

models = [
    dict(
        abbr='intern-s1',
        key='YOUR_API_KEY',
        openai_api_base='YOUR_API_BASE',
        type=OpenAISDK,
        path='internlm/Intern-S1',
        temperature=0.7,
        meta_template=api_meta_template,
        query_per_second=1,
        batch_size=8,
        max_out_len=64000,
        max_seq_len=65536,
        openai_extra_kwargs={
            'top_p': 0.95,
        },
        retry=10,
        extra_body={
            'chat_template_kwargs': {'enable_thinking': True}
        },
        pred_postprocessor=dict(type=extract_non_reasoning_content),
    ),
]