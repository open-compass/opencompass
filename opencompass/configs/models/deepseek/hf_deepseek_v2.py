from opencompass.models import HuggingFaceBaseModel

models = [
    dict(
        type=HuggingFaceBaseModel,
        abbr='deepseek-v2-hf',
        path='deepseek-ai/DeepSeek-V2',
        max_out_len=1024,
        batch_size=4,
        model_kwargs=dict(
            device_map='sequential',
            torch_dtype='torch.bfloat16',
            max_memory={i: '75GB' for i in range(8)},
            attn_implementation='eager'
        ),
        run_cfg=dict(num_gpus=8),
    )
]
