from opencompass.models import HuggingFace

_meta_template = dict(
    round=[
        dict(role='HUMAN', begin='[Round 1]\n\n问：', end='\n\n'),
        dict(role='BOT', begin='答：', end='\n\n', generate=True),
    ],
)

models = [
    dict(
        type=HuggingFace,
        abbr='chatglm2-6b-hf',
        path='THUDM/chatglm2-6b',
        tokenizer_path='THUDM/chatglm2-6b',
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
        max_seq_len=4096,
        batch_size=8,
        run_cfg=dict(num_gpus=1, num_procs=1),
    )
]
