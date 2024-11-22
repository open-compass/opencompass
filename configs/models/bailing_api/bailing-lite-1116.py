from opencompass.models import BailingAPI

models = [
    dict(
        path='Bailing-Lite-1116',
        token='',  # set your key here or in environment variable BAILING_API_KEY
        url='https://bailingchat.alipay.com/chat/completions',
        type=BailingAPI,
        max_out_len=11264,
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
