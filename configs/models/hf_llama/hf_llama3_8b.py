from opencompass.models import HuggingFaceAboveV433Base

models = [
    dict(
        type=HuggingFaceAboveV433Base,
        abbr='llama-3-8b-hf',
        path='meta-llama/Meta-Llama-3-8B',
        max_out_len=1024,
        batch_size=8,
        run_cfg=dict(num_gpus=1),
    )
]
