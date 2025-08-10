from opencompass.models import Gemini

api_meta_template = dict(round=[
    dict(role='HUMAN', api_role='HUMAN'),
    dict(role='BOT', api_role='BOT', generate=True),
], )

models = [
    dict(
        abbr='gemini-1.5-pro',
        type=Gemini,
        path='gemini-1.5-pro',
        key=
        'ENV',  # The key will be obtained from $GEMINI_API_KEY, but you can write down your key here as well
        meta_template=api_meta_template,
        query_per_second=2,
        max_out_len=100,
        max_seq_len=2048,
        batch_size=1,
        temperature=1,
    )
]
