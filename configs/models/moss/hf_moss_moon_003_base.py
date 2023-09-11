from opencompass.models import HuggingFaceCausalLM


models = [
    dict(
        type=HuggingFaceCausalLM,
        abbr='moss-moon-003-base-hf',
        path='fnlp/moss-moon-003-base',
        tokenizer_path='fnlp/moss-moon-003-base',
        tokenizer_kwargs=dict(
            padding_side='left',
            truncation_side='left',
            trust_remote_code=True,
        ),
        max_out_len=100,
        max_seq_len=2048,
        batch_size=8,
        model_kwargs=dict(trust_remote_code=True, device_map='auto', revision='5e406ca0ebbdea11cc3b12aa5932995c692568ac'),
        run_cfg=dict(num_gpus=1, num_procs=1),
    )
]
