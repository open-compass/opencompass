from opencompass.models import HuggingFaceCausalLM

_meta_template = dict(
    round=[
        dict(role='HUMAN', begin='Question:\n', end='\n'),
        dict(role='BOT', begin='Answer:\n', end='\n', generate=True),
    ],
)

models = [
    dict(
        abbr='abel-7b-002',
        type=HuggingFaceCausalLM,
        path='GAIR/Abel-7B-002',
        tokenizer_path='GAIR/Abel-7B-002',
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
        run_cfg=dict(num_gpus=1, num_procs=1),
    )
]
