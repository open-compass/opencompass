from opencompass.models import HuggingFaceAboveV433Chat

models = [
    dict(
        type=HuggingFaceAboveV433Chat,
        abbr='vicuna-7b-v1.3-hf',
        path='lmsys/vicuna-7b-v1.3',
        max_out_len=1024,
        batch_size=8,
        run_cfg=dict(num_gpus=1),
        fastchat_template='vicuna',
    )
]
