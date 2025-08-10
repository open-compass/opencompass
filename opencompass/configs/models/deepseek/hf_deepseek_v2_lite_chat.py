from opencompass.models import HuggingFacewithChatTemplate

models = [
    dict(
        type=HuggingFacewithChatTemplate,
        abbr='deepseek-v2-lite-chat-hf',
        path='deepseek-ai/DeepSeek-V2-Lite-Chat',
        max_out_len=1024,
        batch_size=4,
        model_kwargs=dict(
            device_map='sequential',
            torch_dtype='torch.bfloat16',
            attn_implementation='eager'
        ),
        run_cfg=dict(num_gpus=2),
    )
]
