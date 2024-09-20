from opencompass.models import HuggingFaceBaseModel

models = [
    dict(
        type=HuggingFaceBaseModel,
        abbr='llama-2-70b-hf',
        path='meta-llama/Llama-2-70b-hf',
        max_out_len=1024,
        batch_size=8,
        run_cfg=dict(num_gpus=8),
    )
]
