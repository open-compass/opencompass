from opencompass.models import ClaudeSDK

models = [
    dict(
        type=ClaudeSDK,
        abbr='claude-fable-5',
        path='claude-fable-5',
        key='ENV',
        max_seq_len=1000000,
        max_out_len=131072,
        query_per_second=1,
        retry=10,
        temperature=1.0,
        thinking=dict(type='adaptive', display='summarized'),
        claude_extra_kwargs=dict(
            output_config=dict(effort='max'),
        ),
        batch_size=8,
        max_workers=8,
    )
]
