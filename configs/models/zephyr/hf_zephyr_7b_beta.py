from opencompass.models import HuggingFace

_meta_template = dict(
    round=[
        dict(role="HUMAN", begin='<|user|>\n', end='</s>'),
        dict(role="BOT", begin="<|assistant|>\n", end='</s>', generate=True),
    ],
)

models = [
    dict(
        type=HuggingFace,
        abbr='zephyr-7b-beta-hf',
        path='HuggingFaceH4/zephyr-7b-beta',
        tokenizer_path='HuggingFaceH4/zephyr-7b-beta',
        model_kwargs=dict(
            trust_remote_code=True,
            device_map='auto',
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
        end_str='</s>',
    )
]
