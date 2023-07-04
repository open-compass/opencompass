from opencompass.models import OpenAI

models = [
    dict(abbr='GPT-3.5-turbo',
        type=OpenAI, path='gpt-3.5-turbo', key='sk-xxx',
        max_out_len=2048, max_seq_len=2048, batch_size=1)
]
