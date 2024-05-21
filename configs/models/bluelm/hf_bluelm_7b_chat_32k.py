from opencompass.models import HuggingFaceCausalLM

_meta_template = dict(
    round=[
        dict(role='HUMAN', begin='[|Human|]:'),
        dict(role='BOT', begin='[|AI|]:', generate=True),
    ],
)

models = [
    dict(
        type=HuggingFaceCausalLM,
        abbr='bluelm-7b-chat-32k-hf',
        path='vivo-ai/BlueLM-7B-Chat-32K',
        tokenizer_path='vivo-ai/BlueLM-7B-Chat-32K',
        model_kwargs=dict(
            device_map='auto',
            trust_remote_code=True,
        ),
        tokenizer_kwargs=dict(
            padding_side='left',
            truncation_side='left',
            trust_remote_code=True,
            use_fast=False,
        ),
        meta_template=_meta_template,
        max_out_len=100,
        max_seq_len=4096,
        batch_size=8,
        run_cfg=dict(num_gpus=1, num_procs=1),
    )
]
