from opencompass.models import HuggingFaceCausalLM

_meta_template = dict(
    round=[
        dict(role='HUMAN', begin='', end=''),
        dict(role='BOT', begin='', end='\n\n', generate=True),
    ],
)

models = [
    dict(
        abbr='nanbeige-16b-base-hf',
        type=HuggingFaceCausalLM,
        path='Nanbeige/Nanbeige-16B-Base',
        tokenizer_path='Nanbeige/Nanbeige-16B-Base',
        model_kwargs=dict(
            device_map='auto',
            trust_remote_code=True,
            torch_dtype='auto',
        ),
        tokenizer_kwargs=dict(
            padding_side='right',
            truncation_side='left',
            trust_remote_code=True
        ),
        meta_template=_meta_template,
        batch_padding=False,
        max_out_len=1024,
        max_seq_len=4096,
        batch_size=1,
        run_cfg=dict(num_gpus=1, num_procs=1),
    )
]
