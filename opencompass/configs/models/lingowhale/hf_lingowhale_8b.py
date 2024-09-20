from opencompass.models import HuggingFace


models = [
    dict(
        type=HuggingFace,
        abbr='lingowhale-8b-hf',
        path='deeplang-ai/LingoWhale-8B',
        tokenizer_path='deeplang-ai/LingoWhale-8B',
        model_kwargs=dict(
            trust_remote_code=True,
            device_map='auto',
            torch_dtype='auto',
        ),
        tokenizer_kwargs=dict(
            padding_side='left',
            truncation_side='left',
            trust_remote_code=True,
        ),
        max_out_len=100,
        max_seq_len=2048,
        batch_size=8,
        run_cfg=dict(num_gpus=1, num_procs=1),
    )
]
