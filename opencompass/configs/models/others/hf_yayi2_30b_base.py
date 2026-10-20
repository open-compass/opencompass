from opencompass.models import HuggingFaceCausalLM


models = [
    dict(
        abbr='yayi2-30b-hf',
        type=HuggingFaceCausalLM,
        path='wenge-research/yayi2-30b',
        tokenizer_path='wenge-research/yayi2-30b',
        model_kwargs=dict(
            device_map='auto',
            trust_remote_code=True,
        ),
        tokenizer_kwargs=dict(
            padding_side='left',
            truncation_side='left',
            trust_remote_code=True,
        ),
        max_out_len=100,
        min_out_len=1,
        max_seq_len=2048,
        batch_size=8,
        run_cfg=dict(num_gpus=4, num_procs=1),
    )
]
