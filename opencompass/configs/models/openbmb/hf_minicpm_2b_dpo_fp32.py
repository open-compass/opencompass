from opencompass.models import HuggingFacewithChatTemplate

models = [
    dict(
        type=HuggingFacewithChatTemplate,
        abbr='minicpm-2b-dpo-fp32-hf',
        path='openbmb/MiniCPM-2B-dpo-fp32',
        max_out_len=1024,
        batch_size=8,
        run_cfg=dict(num_gpus=1),
    )
]
