from opencompass.models import HuggingFaceCausalLM

models = [
    dict(
        abbr='orionstar-14b-base-hf',
        type=HuggingFaceCausalLM,
        path='OrionStarAI/Orion-14B-Base',
        tokenizer_path='OrionStarAI/Orion-14B-Base',
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
        run_cfg=dict(num_gpus=2, num_procs=1),
    )
]
