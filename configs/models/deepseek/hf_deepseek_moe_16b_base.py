from opencompass.models import HuggingFaceCausalLM

models = [
    dict(
        type=HuggingFaceCausalLM,
        abbr='deepseek-moe-16b-base-hf',
        path="deepseek-ai/deepseek-moe-16b-base",
        tokenizer_path='deepseek-ai/deepseek-moe-16b-base',
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
        min_out_len=3,
        max_seq_len=2048,
        batch_size=8,
        run_cfg=dict(num_gpus=2, num_procs=1),
    )
]
