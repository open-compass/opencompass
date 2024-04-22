from opencompass.models import HuggingFaceAboveV433Chat

models = [
    dict(
        type=HuggingFaceAboveV433Chat,
        abbr='llama-3-70b-instruct-hf',
        path='meta-llama/Meta-Llama-3-70B-Instruct',
        max_out_len=1024,
        batch_size=8,
        run_cfg=dict(num_gpus=4),
        generation_kwargs={'eos_token_id': [128001, 128009]},
    )
]
