from opencompass.models import HuggingFacewithChatTemplate

models = [
    dict(
        type=HuggingFacewithChatTemplate,
        abbr='MiniCPM3-4B-hf',
        path='openbmb/MiniCPM3-4B',
        max_out_len=1024,
        batch_size=8,
        run_cfg=dict(num_gpus=1),
        model_kwargs=dict(
            torch_dtype='torch.bfloat16',
        ),
    )
]
