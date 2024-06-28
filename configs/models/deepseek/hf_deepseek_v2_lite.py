from opencompass.models import HuggingFaceBaseModel

models = [
    dict(
        type=HuggingFaceBaseModel,
        abbr='deepseek-v2-lite-hf',
        path='deepseek-ai/DeepSeek-V2-Lite',
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
