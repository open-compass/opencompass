from opencompass.models import HuggingFaceCausalLM


_meta_template = dict(
    round=[
        dict(role='HUMAN', begin='### User:\n', end='\n'),
        dict(role='BOT', begin='### Assistant:\n', generate=True),
    ],
)

models = [
    dict(
        abbr='fashiongpt-70b-v11-hf',
        type=HuggingFaceCausalLM,
        path='ICBU-NPU/FashionGPT-70B-V1.1',
        tokenizer_path='ICBU-NPU/FashionGPT-70B-V1.1',
        model_kwargs=dict(
            device_map='auto',
            trust_remote_code=True,
        ),
        tokenizer_kwargs=dict(
            padding_side='left',
            truncation_side='left',
            trust_remote_code=True,
        ),
        meta_template=_meta_template,
        max_out_len=100,
        max_seq_len=2048,
        batch_size=8,
        run_cfg=dict(num_gpus=8, num_procs=1),
    )
]
