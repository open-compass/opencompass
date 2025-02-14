from opencompass.models import HuggingFacewithChatTemplate

models = [
    dict(
        type=HuggingFacewithChatTemplate,
        abbr='internlm3-8b-instruct-hf',
        path='internlm/internlm3-8b-instruct',
        max_out_len=8192,
        batch_size=8,
        run_cfg=dict(num_gpus=1),
    )
]
