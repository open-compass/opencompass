from opencompass.models import HuggingFaceAboveV433Chat

models = [
    dict(
        type=HuggingFaceAboveV433Chat,
        abbr='CodeLlama-70b-Instruct',
        path='codellama/CodeLlama-70b-Instruct-hf',
        max_out_len=1024,
        batch_size=8,
        run_cfg=dict(num_gpus=4),
    )
]
