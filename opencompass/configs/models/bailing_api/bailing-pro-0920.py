from opencompass.models import BailingAPI

api_meta_template = dict(
    round=[
        dict(role='HUMAN', api_role='HUMAN'),
        dict(role='BOT', api_role='BOT', generate=False),
    ],
    reserved_roles=[dict(role='SYSTEM', api_role='SYSTEM')],
)

models = [
    dict(
        path='Bailing-Pro-0920',
        token='',  # set your key here or in environment variable BAILING_API_KEY
        url='https://bailingchat.alipay.com/chat/completions',
        type=BailingAPI,
        meta_template=api_meta_template,
        query_per_second=1,
        max_seq_len=4096,
        batch_size=1,
        generation_kwargs={
            'temperature': 0.4,
            'top_p': 1.0,
            'top_k': -1,
            'n': 1,
            'logprobs': 1,
            'use_beam_search': False,
        },
    ),
]
