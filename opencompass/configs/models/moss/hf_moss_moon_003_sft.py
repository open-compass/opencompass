from opencompass.models import HuggingFaceCausalLM

_meta_template = dict(
    round=[
        dict(role='HUMAN', begin='<|Human|>: ', end='<eoh>\n'),
        dict(role='BOT', begin='<|MOSS|>: ', end='<eom>\n', generate=True),
    ],
)

models = [
    dict(
        type=HuggingFaceCausalLM,
        abbr='moss-moon-003-sft-hf',
        path='fnlp/moss-moon-003-sft',
        tokenizer_path='fnlp/moss-moon-003-sft',
        tokenizer_kwargs=dict(
            padding_side='left',
            truncation_side='left',
            trust_remote_code=True,
        ),
        meta_template=_meta_template,
        max_out_len=100,
        max_seq_len=2048,
        batch_size=8,
        model_kwargs=dict(trust_remote_code=True, device_map='auto', revision='7119d446173035561f40977fb9cb999995bb7517'),
        run_cfg=dict(num_gpus=1, num_procs=1),
    )
]
