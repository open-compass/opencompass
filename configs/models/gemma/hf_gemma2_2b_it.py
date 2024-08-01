from opencompass.models import HuggingFacewithChatTemplate

models = [
    dict(
        type=HuggingFacewithChatTemplate,
        abbr='gemma2-2b-it-hf',
        path='google/gemma-2-2b-it',
        max_out_len=2048,
        batch_size=4,
        run_cfg=dict(num_gpus=1),
        stop_words=['<end_of_turn>'],
        model_kwargs=dict(
            torch_dtype='torch.bfloat16',
        )
    )
]
