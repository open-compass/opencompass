from opencompass.models import HuggingFaceCausalLM


models = [
    dict(
        abbr='mixtral-8x7b-v0.1',
        type=HuggingFaceCausalLM,
        path='mistralai/Mixtral-8x7B-v0.1',
        tokenizer_path='mistralai/Mixtral-8x7B-v0.1',
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
        max_seq_len=2048,
        batch_size=8,
        run_cfg=dict(num_gpus=1, num_procs=1),
    )
]
