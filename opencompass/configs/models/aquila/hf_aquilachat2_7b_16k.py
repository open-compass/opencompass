from opencompass.models import HuggingFaceCausalLM

_meta_template = dict(
    begin='###',
    round=[
        dict(role='HUMAN', begin='Human: ', end='###'),
        dict(role='BOT', begin='Assistant: ', end='</s>', generate=True),
    ],
)

models = [
    dict(
        type=HuggingFaceCausalLM,
        abbr='aquilachat2-7b-16k-hf',
        path='BAAI/AquilaChat2-7B-16K',
        tokenizer_path='BAAI/AquilaChat2-7B-16K',
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
