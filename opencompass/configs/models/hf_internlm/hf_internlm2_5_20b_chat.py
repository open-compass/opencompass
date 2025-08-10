from opencompass.models import HuggingFacewithChatTemplate

models = [
    dict(
        type=HuggingFacewithChatTemplate,
        abbr='internlm2_5-20b-chat-hf',
        path='internlm/internlm2_5-20b-chat',
        max_out_len=2048,
        batch_size=8,
        run_cfg=dict(num_gpus=2),
    )
]
