from opencompass.models import OpenAISDK

models = [
    dict(
        abbr='deepseek_r1_api_sensetime',
        type=OpenAISDK,
        path='DeepSeek-R1',
        key='ENV_SENSETIME',
        openai_api_base='https://api.sensenova.cn/compatible-mode/v1/',
        # TODO: recover qps
        query_per_second=0.1,
        max_out_len=8192,
        max_seq_len=32768,
        batch_size=1,
        retry=30,
        verbose=True,
    ),
]
