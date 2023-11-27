from opencompass.models import HuggingFaceCausalLM

_meta_template = dict(
    round=[
        dict(role='HUMAN', begin='', end=''),
        dict(role='BOT', begin='', end='\n\n', generate=True),
    ],
)

models = [
    dict(
        type=HuggingFaceCausalLM,
        abbr='nanbeige-16b-base-32k-hf',
        path="Nanbeige/Nanbeige-16B-Base-32K",
        tokenizer_path='Nanbeige/Nanbeige-16B-Base-32K',
        model_kwargs=dict(
            device_map='auto',
            trust_remote_code=True,
            torch_dtype='auto',
        ),
        tokenizer_kwargs=dict(
            padding_side='right',
            truncation_side='left',
            trust_remote_code=True,
            use_fast=False,
        ),
        meta_template=_meta_template,
        batch_padding=False,
        max_out_len=1024,
        max_seq_len=8192,
        batch_size=8,
        run_cfg=dict(num_gpus=1, num_procs=1),
    )
]
