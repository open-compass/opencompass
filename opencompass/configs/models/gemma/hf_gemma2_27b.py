from opencompass.models import HuggingFaceBaseModel

models = [
    dict(
        type=HuggingFaceBaseModel,
        abbr='gemma2-27b-hf',
        path='google/gemma-2-27b',
        max_out_len=1024,
        batch_size=4,
        run_cfg=dict(num_gpus=2),
        model_kwargs=dict(
            torch_dtype='torch.bfloat16',
        ),
    )
]
