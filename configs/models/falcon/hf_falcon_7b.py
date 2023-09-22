from opencompass.models import HuggingFaceCausalLM


models = [
    dict(
        type=HuggingFaceCausalLM,
        abbr='falcon-7b-hf',
        path='tiiuae/falcon-7b',
        tokenizer_path='tiiuae/falcon-7b',
        tokenizer_kwargs=dict(
            padding_side='left',
            truncation_side='left',
            trust_remote_code=True,
        ),
        max_out_len=100,
        max_seq_len=2048,
        batch_size=8,
        model_kwargs=dict(trust_remote_code=True, device_map='auto', revision='2f5c3cd4eace6be6c0f12981f377fb35e5bf6ee5'),
        run_cfg=dict(num_gpus=1, num_procs=1),
    )
]
