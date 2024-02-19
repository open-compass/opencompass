from opencompass.models import HuggingFaceCausalLM

models = [
    dict(
        type=HuggingFaceCausalLM,
        abbr='rwkv-5-3b',
        path='RWKV/rwkv-5-world-3b',
        tokenizer_path='RWKV/rwkv-5-world-3b',
        model_kwargs=dict(
            device_map='auto',
            trust_remote_code=True,
        ),
        tokenizer_kwargs=dict(
            padding_side='left',
            truncation_side='left',
            trust_remote_code=True,
            use_fast=False,
        ),
        max_out_len=100,
        max_seq_len=2048,
        batch_padding=True,
        batch_size=16,
        run_cfg=dict(num_gpus=1, num_procs=1),
    )
]
