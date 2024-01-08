from opencompass.models import VLLM


_meta_template = dict(
    begin="<s>",
    round=[
        dict(role="HUMAN", begin='[INST]', end='[/INST]'),
        dict(role="BOT", begin="", end='</s>', generate=True),
    ],
    eos_token_id=2
)

models = [
    dict(
        type=VLLM,
        abbr='mistral-7b-instruct-v0.2-vllm',
        path='mistralai/Mistral-7B-Instruct-v0.2',
        meta_template=_meta_template,
        max_out_len=100,
        max_seq_len=2048,
        batch_size=32,
        generation_kwargs=dict(temperature=0),
        end_str='</s>',
        run_cfg=dict(num_gpus=1, num_procs=1),
    )
]
