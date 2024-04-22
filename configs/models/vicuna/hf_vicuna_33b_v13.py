from opencompass.models import HuggingFaceAboveV433Chat

models = [
    dict(
        type=HuggingFaceAboveV433Chat,
        abbr='vicuna-33b-v1.3-hf',
        path='lmsys/vicuna-33b-v1.3',
        max_out_len=1024,
        batch_size=8,
        run_cfg=dict(num_gpus=2),
        fastchat_template='vicuna',
    )
]
