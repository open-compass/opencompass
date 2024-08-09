from opencompass.models import HuggingFaceBaseModel

models = [
    dict(
        type=HuggingFaceBaseModel,
        abbr='llama-3-70b-hf',
        path='meta-llama/Meta-Llama-3-70B',
        max_out_len=1024,
        batch_size=8,
        run_cfg=dict(num_gpus=8),
    )
]
