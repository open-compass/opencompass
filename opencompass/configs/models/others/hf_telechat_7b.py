from opencompass.models import HuggingFaceCausalLM


_meta_template = dict(
    round=[
        dict(role='HUMAN', begin='<_user>'),
        dict(role='BOT', begin='<_bot>', end='<_end>', generate=True),
    ],
)

models = [
    dict(
        abbr='telechat-7b-hf',
        type=HuggingFaceCausalLM,
        path='Tele-AI/telechat-7B',
        model_kwargs=dict(device_map='auto', trust_remote_code=True),
        tokenizer_kwargs=dict(padding_side='left', truncation_side='left', trust_remote_code=True),
        meta_template=_meta_template,
        max_out_len=100,
        max_seq_len=2048,
        batch_size=8,
        run_cfg=dict(num_gpus=1, num_procs=1),
        end_str='<_end>',
    )
]
