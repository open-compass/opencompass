from opencompass.models import HuggingFacewithChatTemplate

models = [
    dict(
        type=HuggingFacewithChatTemplate,
        abbr='nanbeige2-16b-chat-hf',
        path='Nanbeige/Nanbeige2-16B-Chat',
        max_out_len=1024,
        batch_size=4,
        run_cfg=dict(num_gpus=2),
    )
]
