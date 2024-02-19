from opencompass.models import HuggingFaceCausalLM

models = [
    dict(
        type=HuggingFaceCausalLM,
        abbr='qwen1.5-72b-hf',
        path="Qwen/Qwen1.5-72B",
        tokenizer_path='Qwen/Qwen1.5-72B',
        model_kwargs=dict(
            device_map='auto',
            trust_remote_code=True
        ),
        tokenizer_kwargs=dict(
            padding_side='left',
            truncation_side='left',
            trust_remote_code=True,
            use_fast=False,
        ),
        pad_token_id=151645,
        max_out_len=100,
        max_seq_len=2048,
        batch_size=8,
        run_cfg=dict(num_gpus=4, num_procs=1),
    )
]
