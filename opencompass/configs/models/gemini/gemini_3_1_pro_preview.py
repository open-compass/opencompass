from opencompass.models import GeminiSDK

models = [
    dict(
        type=GeminiSDK,
        abbr='gemini-3.1-pro-preview',
        path='gemini-3.1-pro-preview',
        key='ENV',  # GOOGLE_API_KEY，其次 GEMINI_API_KEY
        max_seq_len=1000000,
        max_out_len=65536,
        query_per_second=1,
        max_workers=8,
        retry=10,
        temperature=1.0,
        thinking=dict(thinking_level='high', include_thoughts=True),
        batch_size=8,
    )
]
