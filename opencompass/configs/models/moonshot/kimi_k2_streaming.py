from opencompass.models import OpenAISDKStreaming

api_meta_template = dict(round=[
    dict(role='HUMAN', api_role='HUMAN'),
    dict(role='BOT', api_role='BOT', generate=True),
], )

models = [
    dict(
        abbr='kimi-k2-0711-preview-streaming',
        type=OpenAISDKStreaming,
        path='kimi-k2-0711-preview',
        key='your-api-key-here',  # Set your API key here
        meta_template=api_meta_template,
        query_per_second=1,
        openai_api_base="https://api.moonshot.cn/v1",
        batch_size=1,
        temperature=1,
        max_seq_len=131072,
        retry=10,
        stream=True,  # 启用流式输出
        verbose=True,  # 启用详细日志，可以看到实时流式输出
        stream_chunk_size=1,  # 流式输出块大小
        ),
] 