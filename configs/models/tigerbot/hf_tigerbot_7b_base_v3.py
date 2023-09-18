from opencompass.models import HuggingFaceCausalLM


models = [
    dict(
        type=HuggingFaceCausalLM,
        abbr='tigerbot-7b-base-v3-hf',
        path='TigerResearch/tigerbot-7b-base',
        tokenizer_path='TigerResearch/tigerbot-7b-base',
        tokenizer_kwargs=dict(
            padding_side='left',
            truncation_side='left',
            trust_remote_code=True,
        ),
        max_out_len=100,
        max_seq_len=2048,
        batch_size=8,
        model_kwargs=dict(trust_remote_code=True, device_map='auto'),
        run_cfg=dict(num_gpus=1, num_procs=1),
    ),
]
