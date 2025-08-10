from opencompass.models import BlueLMAPI

api_meta_template = dict(round=[
    dict(role='HUMAN', api_role='HUMAN'),
    dict(role='BOT', api_role='BOT', generate=True),
], )

models = [
    dict(
        abbr='BlueLM',
        type=BlueLMAPI,
        path='bluelm-2.5',
        key=None,
        batch_size=1,
        meta_template=api_meta_template,
        url = 'http://api-ai.vivo.com.cn/multimodal',
        generation_kwargs={
            'temperature': 0.6,
            'max_tokens': 32768,
            'top_k': 20,
            'top_p': 0.95
        },
    )
]
