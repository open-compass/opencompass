from opencompass.models import HuggingFaceCausalLM


models = [
    dict(
        type=HuggingFaceCausalLM,
        abbr='mpt-7b-hf',
        path='mosaicml/mpt-7b',
        tokenizer_path='mosaicml/mpt-7b',
        tokenizer_kwargs=dict(
            padding_side='left',
            truncation_side='left',
            trust_remote_code=True,
            use_fast=True
        ),
        max_out_len=100,
        max_seq_len=2048,
        batch_size=8,
        model_kwargs=dict(
            device_map='auto',
            trust_remote_code=True,
            max_seq_len=4096,
            revision='68e1a8e0ebb9b30f3c45c1ef6195980f29063ae2',
        ),
        run_cfg=dict(num_gpus=1, num_procs=1),
    )
]
