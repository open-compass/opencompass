from opencompass.models import HuggingFaceCausalLM

_meta_template = dict(
    round=[
        dict(role="HUMAN", begin='Question: ', end='\n\n'),
        dict(role="BOT", begin="Answer:", end='', generate=True),
    ],
)

models = [
    dict(
        type=HuggingFaceCausalLM,
        abbr='rwkv-5-3b',
        path="RWKV/rwkv-5-world-3b",
        tokenizer_path='RWKV/rwkv-5-world-3b',
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
        max_out_len=4096,
        max_seq_len=4096,
        batch_size=1,
        meta_template=_meta_template,
        run_cfg=dict(num_gpus=1, num_procs=1),
    )
]
