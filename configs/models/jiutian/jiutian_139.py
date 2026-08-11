from opencompass.models import JiutianApi

models = [
    dict(
        abbr='JIUTIAN-13.9B',
        type=JiutianApi,
        path='jiutian-cm',
        appcode='',
        url='https://jiutian.10086.cn/kunlun/ingress/api/h3t-f9c8f9/fae3164b494b4d97b7011c839013c912/ai-7f03963dae10471bb42b6a763a875a68/service-d4cc837d3fe34656a7c0eebd6cec8311/v1/chat/completions',
        max_seq_len=8192,
        max_out_len=4096,
        batch_size=1,
        max_tokens=512,
        model_id='jiutian-cm'
    )
]
