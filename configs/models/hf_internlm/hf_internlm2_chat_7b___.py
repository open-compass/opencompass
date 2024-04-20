from opencompass.models import HuggingFaceAboveV433Chat


models = [
    dict(
        type=HuggingFaceAboveV433Chat,
        path="internlm/internlm2-chat-7b",
        max_out_len=1024,
        batch_size=32,
        run_cfg=dict(num_gpus=1),
        generation_kwargs={"eos_token_id": [2, 92542], "do_sample": True},
    )
]
