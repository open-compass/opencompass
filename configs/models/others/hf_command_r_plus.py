from opencompass.models import HuggingFaceAboveV433Chat

models = [
    dict(
        type=HuggingFaceAboveV433Chat,
        abbr='command-r-plus-hf',
        path='CohereForAI/c4ai-command-r-plus',
        max_out_len=1024,
        batch_size=8,
        run_cfg=dict(num_gpus=8),
    )
]
