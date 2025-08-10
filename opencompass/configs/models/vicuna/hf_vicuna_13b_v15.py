from opencompass.models import HuggingFacewithChatTemplate

models = [
    dict(
        type=HuggingFacewithChatTemplate,
        abbr='vicuna-13b-v1.5-hf',
        path='lmsys/vicuna-13b-v1.5',
        max_out_len=1024,
        batch_size=8,
        run_cfg=dict(num_gpus=1),
        fastchat_template='vicuna',
    )
]
